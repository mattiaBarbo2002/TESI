import os
import zipfile
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import digitalhub as dh
import matplotlib
import random
import json
import zipfile
import contextlib
import io
import sys
import re
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from digitalhub_runtime_python import handler

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA 
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict

from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_2d
from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_attn
from floodlib.loader import MoCo2encodersLoader
from floodlib.builder import MoCo2encoders_2d
from floodlib.builder import MoCo2encoders_attn


matplotlib.use("Agg") 


# notebook -> ad_cosine

@handler()
def ad_cosine(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
    attn: bool = False,
    hidden_channels_dim: int = 64,
    pool_grid: int = 1,
    moco_dim: int = 128,
    moco_k: int = 4096,
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    n_std: float = 1.0,
    batch_size: int = 32,
    workers: int = 0,
    min_flood_area: float = 10.0, 
    old_vectors: str = "",
    moco_attn: bool = False                
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
  
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    # lettura ids serie anomale
    print(f"Lettura serie anomale: {dataset_anomalies}", flush=True)
    anom_path = project_data.get_artifact(f"{dataset_anomalies}").download(f"/data/{dataset_anomalies}")
    ids_per_sensor_anom = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(anom_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_anom[sensor] = ids
 
    anom_ids = ids_per_sensor_anom["SAR"] & ids_per_sensor_anom["OPT"]
    num_thr_series = len(anom_ids)
    n_anom_tot = len(anom_ids)
 
    # filtro superficie alluvione
    if min_flood_area > 0:
        try:
            mask_parts = sorted(glob(os.path.join(anom_path, "MASK", "part*.zip")))
 
            flood_area = {}
            for part_path in mask_parts:
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('.npy') and name[:-4] in anom_ids:
                            mask = np.load(io.BytesIO(z.read(name)))
                            flood_area[name[:-4]] = 100.0 * float((mask == 2).sum()) / mask.size
 
            anom_ids = {ID for ID in anom_ids if flood_area.get(ID, 0.0) >= min_flood_area}
            print(f"OK -> flood >= {min_flood_area:g}%: {len(anom_ids)} serie su {n_anom_tot}", flush=True)
 
        except Exception as e:
            print(f"EXC -> flood_ {e}", flush=True)
 
    anom_ids = sorted(anom_ids)
    print(f"OK -> {len(anom_ids)} serie anomale", "\n", flush=True)
 
 
    # lettura serie normali
    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(f"/data/Floods_{dataset_standard}_crop_norm")
 
    # import ids lits from moco training
    if moco_attn:
        p = project_work.get_artifact(f"moco_attn_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
    else:
        p = project_work.get_artifact(f"moco_2D_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)

    with open(p, 'r') as f:
        test_ids = json.load(f)
    print(f"OK -> liste normali caricate: {len(test_ids)} test", flush=True)
    num_thr_series = len(test_ids) - num_thr_series
 
    random.seed(0)
    shuffled_test_ids = random.sample(test_ids, len(test_ids)) 
 
    ids_per_sensor_normal = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(normal_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_normal[sensor] = ids
 
    normal_ids = ids_per_sensor_normal["SAR"] & ids_per_sensor_normal["OPT"]
 
    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID in normal_ids]
    normal_thr_ids = shuffled_test_ids[:num_thr_series]                                             # thr 
    normal_test_ids = shuffled_test_ids[num_thr_series:(num_thr_series+len(anom_ids))]              # anomaly detection 
 
 
    print(f"OK -> {len(test_ids)} serie normali tot, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali")
 
    if old_vectors:
        print(f"Upload embedding: anomaly_results_cos_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_cos_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            z.extract("embeddings.npz", "/data/old_vectors")
            old_results = json.loads(z.read("anomaly_results.json"))
 
        for key, value in (("moco_version", moco_version), ("dataset_anomalies", dataset_anomalies)):
            if old_results.get(key) != value:
                raise ValueError(f"old_vectors={old_vectors} ha {key}={old_results.get(key)}, questo job ha {value}")
 
        old = np.load("/data/old_vectors/embeddings.npz")
        lookup = {}
        for g in ("fit", "normal", "anom"):
            for i, ID in enumerate(old[f"{g}_ids"]):
                lookup[str(ID)] = (old[f"{g}_q"][i], old[f"{g}_k"][i])
 
        def take(ids, nome):
            missing = [ID for ID in ids if ID not in lookup]
            if missing:
                raise ValueError(f"{len(missing)} serie {nome} non presenti in {old_vectors} (es. {missing[:3]})")
            return np.stack([lookup[ID][0] for ID in ids]), np.stack([lookup[ID][1] for ID in ids])
 
        q_thr, k_thr = take(normal_thr_ids, "fit")
        q_normal, k_normal = take(normal_test_ids, "normali test")
        q_anom, k_anom = take(anom_ids, "anomale")
        print(f"OK -> embedding caricatu: {len(q_thr)} fit, {len(q_normal)} normali, {len(q_anom)} anomale", "\n", flush=True)
 
    else:
 
        print("calcolo vettori MoCo", flush = True)
 
        # estrazione serie normal_ids, thr_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"
 
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_thr_ids) | set(normal_test_ids)}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)
 
        # caricamento pesi moco
        try:
            if moco_attn:
                encoderSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)
                encoderOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)

                model = MoCo2encoders_attn(
                    base_encoder_q=encoderOPT.encoder,
                    base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    symmetric=False, 
                    hidden_channels_dim=hidden_channels_dim,
                    pool_grid=pool_grid,
                    simple_proj=False,
                    attn=attn
                ).to(device)

                w_path = project_work.get_artifact(f"moco_attn_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)

            else:
                encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
                encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)

                model = MoCo2encoders_2d(
                    base_encoder_q=encoderOPT.encoder,
                    base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    symmetric=False, device=device,
                    hidden_channels_dim=hidden_channels_dim,
                    simple_proj=simple_proj,
                    pool_grid=pool_grid,
                    attn=attn
                ).to(device)

                w_path = project_work.get_artifact(f"moco_2D_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)               

            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("OK -> modello MoCo caricato",  "\n", flush=True)
 
        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
 
        # embedding moco pesi congelati
        # 2 matrici: rows = serie, cols = dim
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
 
            out_q, out_k = [], []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)
                    _, q, k = model.contrastive_loss(im_q, im_k)
                    out_q.append(q.float().flatten(1).cpu())
                    out_k.append(k.float().flatten(1).cpu())
            return torch.cat(out_q).numpy(), torch.cat(out_k).numpy()
 
        # embedding serie di thr 
        print("Embedding MoCo: serie soglia", flush=True)
        q_thr, k_thr = embed(normal_thr_ids)
        print(f"OK -> soglia calcolata su {len(q_thr)} serie", "\n", flush=True)
 
        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        q_normal, k_normal = embed(normal_test_ids)
        print(f"OK -> normali test: {len(q_normal)} serie, q {q_normal.shape[1]} + k {k_normal.shape[1]} = {q_normal.shape[1] + k_normal.shape[1]} dim", "\n", flush=True)
 
        # pulizia cartelle
        for d in ("/data/cache_SAR", "/data/cache_OPT"):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
 
        # estrazione serie anomale
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_anom_series = {f"{ID}.npy" for ID in anom_ids}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(anom_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_anom_series:
                            z.extract(name, cache_dir)
 
        # embedding serie anomale
        print("Embedding MoCo: serie anomale", flush=True)
        q_anom, k_anom = embed(anom_ids)
        print(f"OK -> anomale: {len(q_anom)} serie, q {q_anom.shape[1]} + k {k_anom.shape[1]} = {q_anom.shape[1] + k_anom.shape[1]} dim", "\n", flush=True)
 
    # cosine similarity
    cos_sim_thr = (q_thr * k_thr).sum(axis=1)
    cos_sim_normal = (q_normal * k_normal).sum(axis=1)
    cos_sim_anom = (q_anom * k_anom).sum(axis=1)
 
    # soglia: media +- n_std * std sulle serie soglia 
    thr_mean = float(cos_sim_thr.mean())
    thr_std = float(cos_sim_thr.std())
    thr_high = thr_mean + (n_std * thr_std)
    thr_low = thr_mean - (n_std * thr_std)
 
    scores = np.concatenate([cos_sim_normal, cos_sim_anom])
    label = np.concatenate([np.zeros(len(cos_sim_normal)), np.ones(len(cos_sim_anom))])
 
    auc_high = float(roc_auc_score(label, scores))          
    auc_low = 1.0 - auc_high                                
 
    tests = {
        f"cos_sim > m+{n_std:g}*std": (scores > thr_high, {"soglia": thr_high}),
        f"cos_sim < m-{n_std:g}*std": (scores < thr_low, {"soglia": thr_low}),
        f"cos_sim fuori da m+-{n_std:g}*std": ((scores > thr_high) | (scores < thr_low), {"soglia_low": thr_low, "soglia_high": thr_high}),
    }
 
    rules = {}
    for key, (value, soglie) in tests.items():
        tp = int((value & (label == 1)).sum())
        fp = int((value & (label == 0)).sum())
        fn = int((~value & (label == 1)).sum())
        tn = int((~value & (label == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn)
 
        rules[key] = {
            **soglie, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp), "REC": rec, "PREC": prec,
            "F1": 2 * prec * rec / (prec + rec) if tp > 0 else 0.0,
        }
        print(f"{key}: TP {tp} FP {fp} FN {fn} TN {tn} | FA {fp/(fp+tn):.3f} REC {rec:.3f} PREC {prec:.3f}", flush=True)
 
    # salvataggio e risultati 
    out_dir = f"/data/anomaly_results_cos_{job_name}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
 
    # punteggi
    df = pd.DataFrame({
        "ID": normal_test_ids + anom_ids,
        "label": [0] * len(cos_sim_normal) + [1] * len(cos_sim_anom),
        "score": scores.astype(np.float64),
    })
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)

    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        fit_ids=np.array(normal_thr_ids), fit_q=q_thr, fit_k=k_thr,
        normal_ids=np.array(normal_test_ids), normal_q=q_normal, normal_k=k_normal,
        anom_ids=np.array(anom_ids), anom_q=q_anom, anom_k=k_anom,
    )
 
    # riassunto
    results = {
        "job_name": job_name, "metodo": "cosine_similarity", "dataset_anomalies": dataset_anomalies,
        "moco_version": moco_version, "min_flood_area": min_flood_area,
        "n_fit": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
        "thr": {"mean": thr_mean, "std": thr_std, "n_std": n_std, "thr_high": thr_high, "thr_low": thr_low},
        "auroc_cos_sim_high_anom": auc_high,
        "auroc_cos_sim_low_anom": auc_low,
        "rules": rules,
    }
    with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)
 
    shutil.make_archive(out_dir, "zip", out_dir)
    project_work.log_artifact(name=f"anomaly_results_cos_{job_name}", source=f"{out_dir}.zip", kind="artifact")
 
    return "TERMINATO -> anomaly detection cosine"





@handler()
def ad_bin(
    job_name: str = "nome_job",
    old_vectors: str = "",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
    attn: bool = False,
    hidden_channels_dim: int = 64,
    pool_grid: int = 1,
    moco_dim: int = 128,
    moco_k: int = 4096,
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    standardize: bool = True,                       
    batch_size: int = 32,
    workers: int = 0,
    min_flood_area: float = 10.0,                    
    k_bin: int = 1,         
    moco_attn: bool = False                         
):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
  
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    # lettura ids serie anomale
    print(f"Lettura serie anomale: {dataset_anomalies}", flush=True)
    anom_path = project_data.get_artifact(f"{dataset_anomalies}").download(f"/data/{dataset_anomalies}")
    ids_per_sensor_anom = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(anom_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_anom[sensor] = ids
 
    anom_ids = ids_per_sensor_anom["SAR"] & ids_per_sensor_anom["OPT"]
    num_fit_series = len(anom_ids)
 
    # filtro superficie alluvione
    if min_flood_area > 0:
        try:
            mask_parts = sorted(glob(os.path.join(anom_path, "MASK", "part*.zip")))
 
            flood_area = {}
            for part_path in mask_parts:
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('.npy') and name[:-4] in anom_ids:
                            mask = np.load(io.BytesIO(z.read(name)))
                            flood_area[name[:-4]] = 100.0 * float((mask == 2).sum()) / mask.size
 
            anom_ids = {ID for ID in anom_ids if flood_area.get(ID, 0.0) >= min_flood_area}
            print(f"OK -> flood >= {min_flood_area:g}%: {len(anom_ids)} serie su {len(anom_ids)}", flush=True)
 
        except Exception as e:
            print(f"EXC -> flood_ {e}", flush=True)
 
    anom_ids = sorted(anom_ids)
    print(f"OK -> {len(anom_ids)} serie anomale", "\n", flush=True)
 
 
    # lettura serie normali
    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(f"/data/Floods_{dataset_standard}_crop_norm")
 
    # import ids lits from moco training
    if moco_attn:
        p = project_work.get_artifact(f"moco_attn_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
    else:
        p = project_work.get_artifact(f"moco_2D_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
    
    with open(p, 'r') as f:
        test_ids = json.load(f)
    print(f"OK -> liste normali caricate: {len(test_ids)} test", flush=True)
    num_fit_series = len(test_ids) - num_fit_series
 
    random.seed(0)
    shuffled_test_ids = random.sample(test_ids, len(test_ids)) 
 
    ids_per_sensor_normal = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(normal_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_normal[sensor] = ids
 
    normal_ids = ids_per_sensor_normal["SAR"] & ids_per_sensor_normal["OPT"]
 
    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID in normal_ids]
 
    # ref: rima serie evento
    id_pattern = re.compile(r"^(?P<event>.+)_(?P<zone>\d{1,2})[A-Z]{3}_x(?P<x>-?\d+)_y(?P<y>-?\d+)$")
    eventi_visti, ref_ids = set(), []
    for ID in shuffled_test_ids:                                   
        m = id_pattern.match(ID)
        if m is None or m.group("event") in eventi_visti:          
            continue
        eventi_visti.add(m.group("event"))
        ref_ids.append(ID)

    ref_set = set(ref_ids)      
    print(f"OK -> {len(ref_ids)} riferimenti (uno per evento)", flush=True)

    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID not in ref_set] 
    normal_fit_ids = shuffled_test_ids[:num_fit_series]                                             # fit -> otsu
    normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]              # anomaly  
 
    print(f"OK -> {len(test_ids)} serie normali tot, {len(normal_fit_ids)} serie fit, {len(normal_test_ids)} serie test normali")
 
    if old_vectors:
        print(f"Upload embedding: anomaly_results_cos_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_cos_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            z.extract("embeddings.npz", "/data/old_vectors")
            old_results = json.loads(z.read("anomaly_results.json"))
 
        for key, value in (("moco_version", moco_version), ("dataset_anomalies", dataset_anomalies)):
            if old_results.get(key) != value:
                raise ValueError(f"old_vectors={old_vectors} ha {key}={old_results.get(key)}, questo job ha {value}")
 
        old = np.load("/data/old_vectors/embeddings.npz")
        lookup = {}
        for g in ("fit", "normal", "anom"):
            for i, ID in enumerate(old[f"{g}_ids"]):
                lookup[str(ID)] = (old[f"{g}_q"][i], old[f"{g}_k"][i])
 
        def take(ids, nome):
            missing = [ID for ID in ids if ID not in lookup]
            if missing:
                raise ValueError(f"{len(missing)} serie {nome} non presenti in {old_vectors} (es. {missing[:3]})")
            return np.stack([lookup[ID][0] for ID in ids]), np.stack([lookup[ID][1] for ID in ids])
 
        q_fit, k_fit = take(normal_fit_ids, "fit")
        q_normal, k_normal = take(normal_test_ids, "normali test")
        q_anom, k_anom = take(anom_ids, "anomale")
        q_ref, k_ref = take(ref_ids, "ref")
        print(f"OK -> embedding caricatu: {len(q_fit)} fit, {len(q_ref)} ref, {len(q_normal)} normali, {len(q_anom)} anomale", "\n", flush=True)
 
    else:
 
        print("calcolo vettori MoCo", flush = True)
 
        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"
 
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_test_ids) | set(ref_ids)}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)
 
        # caricamento pesi moco
        try:
            if moco_attn:
                encoderSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)
                encoderOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)

                model = MoCo2encoders_attn(
                    base_encoder_q=encoderOPT.encoder,
                    base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    symmetric=False, 
                    hidden_channels_dim=hidden_channels_dim,
                    pool_grid=pool_grid,
                    simple_proj=False,
                    attn=attn
                ).to(device)

                w_path = project_work.get_artifact(f"moco_attn_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)

            else:
                encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
                encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)

                model = MoCo2encoders_2d(
                    base_encoder_q=encoderOPT.encoder,
                    base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    symmetric=False, device=device,
                    hidden_channels_dim=hidden_channels_dim,
                    simple_proj=simple_proj,
                    pool_grid=pool_grid,
                    attn=attn
                ).to(device)

                w_path = project_work.get_artifact(f"moco_2D_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)               
            
            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("OK -> modello MoCo caricato",  "\n", flush=True)
 
        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
 
        # embedding moco: pesi congelati
        # 2 matrici: rows = serie, cols = dim
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
 
            out_q, out_k = [], []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)
                    _, q, k = model.contrastive_loss(im_q, im_k)
                    out_q.append(q.float().flatten(1).cpu())
                    out_k.append(k.float().flatten(1).cpu())
            return torch.cat(out_q).numpy(), torch.cat(out_k).numpy()
 
        # embedding serie di fit 
        print("Embedding MoCo: serie di fit pac", flush=True)
        q_fit, k_fit = embed(normal_fit_ids)
        print(f"OK -> normali fit: {len(q_fit)} serie, q {q_fit.shape[1]} + k {k_fit.shape[1]} = {q_fit.shape[1] + k_fit.shape[1]} dim", "\n", flush=True)
 
        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        q_normal, k_normal = embed(normal_test_ids)
        print(f"OK -> normali test: {len(q_normal)} serie, q {q_normal.shape[1]} + k {k_normal.shape[1]} = {q_normal.shape[1] + k_normal.shape[1]} dim", "\n", flush=True)
 
        # embedding serie di riferimento
        print("Embedding MoCo: serie di riferimento", flush=True)
        q_ref, k_ref = embed(ref_ids)
        print(f"OK -> riferimento: {len(q_ref)} serie, q {q_ref.shape[1]} + k {k_ref.shape[1]} = {q_ref.shape[1] + k_ref.shape[1]} dim", "\n", flush=True)
 
        # pulizia cartelle
        for d in ("/data/cache_SAR", "/data/cache_OPT"):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
 
        # estrazione serie anomale
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_anom_series = {f"{ID}.npy" for ID in anom_ids}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(anom_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_anom_series:
                            z.extract(name, cache_dir)
 
        # embedding serie anomale
        print("Embedding MoCo: serie anomale", flush=True)
        q_anom, k_anom = embed(anom_ids)
        print(f"OK -> anomale: {len(q_anom)} serie, q {q_anom.shape[1]} + k {k_anom.shape[1]} = {q_anom.shape[1] + k_anom.shape[1]} dim", "\n", flush=True)
 
    # concatenazione
    z_fit = np.concatenate([q_fit, k_fit], axis=1)
    z_normal = np.concatenate([q_normal, k_normal], axis=1)
    z_anom = np.concatenate([q_anom, k_anom],axis=1)
    z_ref = np.concatenate([q_ref, k_ref], axis=1)
 
    if standardize:                                   
        scaler = StandardScaler().fit(z_fit)
        z_fit, z_normal, z_anom, z_ref = scaler.transform(z_fit), scaler.transform(z_normal), scaler.transform(z_anom), scaler.transform(z_ref)
 
    # binarizzazione
    def otsu(x, bins=256):
        counts, edges = np.histogram(x, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        w_low = np.cumsum(counts)                                        
        w_high = w_low[-1] - w_low                                       
        s_low = np.cumsum(counts * centers)
        mean_low = s_low / np.maximum(w_low, 1)
        mean_high = (s_low[-1] - s_low) / np.maximum(w_high, 1)
        between = w_low * w_high * (mean_low - mean_high) ** 2           
        return centers[np.argmax(between)]
 
    soglie = np.array([otsu(z_fit[:, j]) for j in range(z_fit.shape[1])])
    b_fit = (z_fit > soglie).astype(np.uint8)                            
    b_normal = (z_normal > soglie).astype(np.uint8)
    b_anom = (z_anom > soglie).astype(np.uint8)
    b_ref = (z_ref > soglie).astype(np.uint8)
 
    p1 = b_fit.mean(axis=0)                                              
    print(f"OK -> otsu: {b_fit.shape[1]} dim | frac 1 fit: min {p1.min():.2f}, mediana {np.median(p1):.2f}, max {p1.max():.2f}", flush=True)
    print(f"frac 1 tot: fit {b_fit.mean():.3f}, normali {b_normal.mean():.3f}, anomale {b_anom.mean():.3f}", "\n", flush=True)
 
    # distanza di Hamming dai riferimenti
    if k_bin > len(b_ref):
        raise ValueError(f"k_bin = {k_bin} -> ({len(b_ref)})")
 
    def hamming(A, R):                                                   
        A = A.astype(np.float32); R = R.astype(np.float32)
        return A @ (1 - R).T + (1 - A) @ R.T
 
    def punteggio(A):                                                    
        return np.sort(hamming(A, b_ref), axis=1)[:, k_bin - 1]
 
    res_normal = punteggio(b_normal)
    res_anom = punteggio(b_anom)
    print(f"distanza -> normali {res_normal.mean():.1f} ± {res_normal.std():.1f} | anomale {res_anom.mean():.1f} ± {res_anom.std():.1f}", "\n", flush=True)
 
    # salvataggio e risultati
    out_dir = f"/data/anomaly_results_bin_{job_name}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
 
    # punteggi
    score_normal = -res_normal.astype(np.float64)
    score_anom = -res_anom.astype(np.float64)
    auroc = roc_auc_score(np.r_[np.zeros(len(res_normal)), np.ones(len(res_anom))], np.r_[res_normal, res_anom])
    print(f"auroc: {auroc:.4f}", "\n", flush=True)
 
    df = pd.DataFrame({
        "ID": normal_test_ids + anom_ids,
        "label": [0] * len(score_normal) + [1] * len(score_anom),
        "score": np.concatenate([score_normal, score_anom]),
    })
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
 
    # soglia
    false_allarms = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    normali = df[df.label == 0].sample(frac=1, random_state=0)
    meta = len(normali) // 2
    thr_scores = normali.score.values[:meta]
    eval_norm = normali.score.values[meta:]
    anom = df[df.label == 1].score.values
 
    rules = {}
    for fa in false_allarms:
        thr = float(np.quantile(thr_scores, fa))
        tp = int((anom < thr).sum()); fn = len(anom) - tp
        fp = int((eval_norm < thr).sum()); tn = len(eval_norm) - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        rules[f"bin score < soglia (FA {fa:.0%})"] = {
            "soglia": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp), "REC": rec, "PREC": prec,
            "F1": 2 * prec * rec / (prec + rec) if tp > 0 else 0.0,
        }
        print(f"FA {fa:.0%}: soglia {thr:.4f} | TP {tp} FP {fp} FN {fn} TN {tn} | FA {fp/(fp+tn):.3f} REC {rec:.3f} PREC {prec:.3f}", flush=True)
 
    # vettori
    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        fit_ids=np.array(normal_fit_ids), fit_q=q_fit, fit_k=k_fit,
        normal_ids=np.array(normal_test_ids), normal_q=q_normal, normal_k=k_normal,
        anom_ids=np.array(anom_ids), anom_q=q_anom, anom_k=k_anom,
        ref_ids=np.array(ref_ids), ref_q=q_ref, ref_k=k_ref,
        soglie_otsu=soglie,
    )
 
    # riassunto
    results = {
        "job_name": job_name, "metodo": "bin", "dataset_anomalies": dataset_anomalies,
        "moco_version": moco_version, "min_flood_area": min_flood_area, "standardize": standardize,
        "k_bin": int(k_bin), "n_ref": len(ref_ids), "n_eventi": len(eventi_visti),
        "n_fit": len(normal_fit_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
        "auroc": float(auroc),
        "rules": rules,
    }
    with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
        json.dump(results, f, indent=2)
 
    shutil.make_archive(out_dir, "zip", out_dir)
    project_work.log_artifact(name=f"anomaly_results_bin_{job_name}", source=f"{out_dir}.zip", kind="artifact")
 
    return "TERMINATO -> anomaly detection bin"

# 

@handler()
def ad_ocsvm(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
    attn: bool = False,
    hidden_channels_dim: int = 64,
    pool_grid: int = 1,
    moco_dim: int = 128,
    moco_k: int = 4096,
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    ocsvm_nu: float = 0.05,                         
    ocsvm_gamma: str = "scale",                     
    standardize: bool = True,                       
    batch_size: int = 32,
    workers: int = 0,
    min_flood_area: float = 10.0,     
    old_vectors: str = "",
    n_series_thr: int = 500,
    moco_attn: bool = False,                                
):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
  
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    # lettura ids serie anomale
    print(f"Lettura serie anomale: {dataset_anomalies}", flush=True)
    anom_path = project_data.get_artifact(f"{dataset_anomalies}").download(f"/data/{dataset_anomalies}")
    ids_per_sensor_anom = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(anom_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_anom[sensor] = ids
 
    anom_ids = ids_per_sensor_anom["SAR"] & ids_per_sensor_anom["OPT"]
    num_fit_series = len(anom_ids)
    n_anom_tot = len(anom_ids)
 
    # filtro superficie alluvione
    if min_flood_area > 0:
        try:
            mask_parts = sorted(glob(os.path.join(anom_path, "MASK", "part*.zip")))
 
            flood_area = {}
            for part_path in mask_parts:
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('.npy') and name[:-4] in anom_ids:
                            mask = np.load(io.BytesIO(z.read(name)))
                            flood_area[name[:-4]] = 100.0 * float((mask == 2).sum()) / mask.size
 
            anom_ids = {ID for ID in anom_ids if flood_area.get(ID, 0.0) >= min_flood_area}
            print(f"OK -> flood >= {min_flood_area:g}%: {len(anom_ids)} serie su {n_anom_tot}", flush=True)
 
        except Exception as e:
            print(f"EXC -> flood: {e}", flush=True)
 
    anom_ids = sorted(anom_ids)
    print(f"OK -> {len(anom_ids)} serie anomale", "\n", flush=True)
 
 
    # lettura serie normali
    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(f"/data/Floods_{dataset_standard}_crop_norm")
 
    # import ids lits from moco training
    if moco_attn:
        p = project_work.get_artifact(f"moco_attn_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
    else:
        p = project_work.get_artifact(f"moco_2D_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
    
    with open(p, 'r') as f:
        test_ids = json.load(f)
    print(f"OK -> liste normali caricate: {len(test_ids)} test", flush=True)
    num_fit_series = len(test_ids) - num_fit_series
 
    random.seed(0)
    shuffled_test_ids = random.sample(test_ids, len(test_ids)) 
 
    ids_per_sensor_normal = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(normal_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_normal[sensor] = ids
 
    normal_ids = ids_per_sensor_normal["SAR"] & ids_per_sensor_normal["OPT"]
 
    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID in normal_ids]

    normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]                            # fit -> iperpiano
    normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]              # calcolo soglie
    normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]             # anomaly detection            
 
 
    print(f"OK -> {len(test_ids)} serie normali tot, {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali")
 
    if old_vectors:
        print(f"Upload embedding: anomaly_results_ocsvm_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_ocsvm_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            z.extract("embeddings.npz", "/data/old_vectors")
            old_results = json.loads(z.read("anomaly_results.json"))
 
        for key, value in (("moco_version", moco_version), ("dataset_anomalies", dataset_anomalies)):
            if old_results.get(key) != value:
                raise ValueError(f"old_vectors={old_vectors} ha {key}={old_results.get(key)}, questo job ha {value}")
 
        old = np.load("/data/old_vectors/embeddings.npz")
        lookup = {}
        for g in ("fit", "thr", "normal", "anom"):
            for i, ID in enumerate(old[f"{g}_ids"]):
                lookup[str(ID)] = (old[f"{g}_q"][i], old[f"{g}_k"][i])
 
        def take(ids, nome):
            missing = [ID for ID in ids if ID not in lookup]
            if missing:
                raise ValueError(f"{len(missing)} serie {nome} non presenti in {old_vectors} (es. {missing[:3]})")
            return np.stack([lookup[ID][0] for ID in ids]), np.stack([lookup[ID][1] for ID in ids])
 
        q_fit, k_fit = take(normal_fit_ids, "fit")
        q_thr, k_thr = take(normal_thr_ids, "thr")
        q_normal, k_normal = take(normal_test_ids, "normali test")
        q_anom, k_anom = take(anom_ids, "anomale")
        print(f"OK -> embedding caricatu: {len(q_fit)} fit,  {len(q_thr)} thr, {len(q_normal)} normali, {len(q_anom)} anomale", "\n", flush=True)
 
    else:
 
        print("calcolo vettori MoCo", flush = True)
 
        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"
 
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids)}       
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)
 
        # caricamento pesi moco
        try:
            if moco_attn:
                encoderSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)
                encoderOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)

                model = MoCo2encoders_attn(
                    base_encoder_q=encoderOPT.encoder,
                    base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    symmetric=False, 
                    hidden_channels_dim=hidden_channels_dim,
                    pool_grid=pool_grid,
                    simple_proj=False,
                    attn=attn
                ).to(device)

                w_path = project_work.get_artifact(f"moco_attn_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)

            else:
                encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
                encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)

                model = MoCo2encoders_2d(
                    base_encoder_q=encoderOPT.encoder,
                    base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    symmetric=False, device=device,
                    hidden_channels_dim=hidden_channels_dim,
                    simple_proj=simple_proj,
                    pool_grid=pool_grid,
                    attn=attn
                ).to(device)

                w_path = project_work.get_artifact(f"moco_2D_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)               
            
            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("OK -> modello MoCo caricato",  "\n", flush=True)
 
        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
 
        # embedding moco: pesi congelati
        # 2 matrici: rows = serie, cols = dim
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
 
            out_q, out_k = [], []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)

                    #_, q, k = model.contrastive_loss(im_q, im_k)
                    q = model.encoder_q(im_q)
                    k = model.encoder_k(im_k)
                    if model.simple_proj:
                        q = model.pool_q(q).flatten(1)
                        k = model.pool_k(k).flatten(1)
                    else:
                        q = model.pool_q(model.conv_q(q)).flatten(1)
                        k = model.pool_k(model.conv_k(k)).flatten(1)
                    q = model.proj_q(q)
                    k = model.proj_k(k)
                    
                    out_q.append(q.float().flatten(1).cpu())
                    out_k.append(k.float().flatten(1).cpu())
            return torch.cat(out_q).numpy(), torch.cat(out_k).numpy()
 
        # embedding serie di fit 
        print("Embedding MoCo: serie di fit ocsvm", flush=True)
        q_fit, k_fit = embed(normal_fit_ids)
        print(f"OK -> normali fit: {len(q_fit)} serie, q {q_fit.shape[1]} + k {k_fit.shape[1]} = {q_fit.shape[1] + k_fit.shape[1]} dim", "\n", flush=True)

        # embedding serie thr 
        print("Embedding MoCo: serie thr ocsvm", flush=True)
        q_thr, k_thr = embed(normal_thr_ids)
        print(f"OK -> normali thr: {len(q_thr)} serie, q {q_thr.shape[1]} + k {k_thr.shape[1]} = {q_thr.shape[1] + k_thr.shape[1]} dim", "\n", flush=True)
 
        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        q_normal, k_normal = embed(normal_test_ids)
        print(f"OK -> normali test: {len(q_normal)} serie, q {q_normal.shape[1]} + k {k_normal.shape[1]} = {q_normal.shape[1] + k_normal.shape[1]} dim", "\n", flush=True)
 
        # pulizia cartelle
        for d in ("/data/cache_SAR", "/data/cache_OPT"):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
 
        # estrazione serie anomale
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_anom_series = {f"{ID}.npy" for ID in anom_ids}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(anom_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_anom_series:
                            z.extract(name, cache_dir)
 
        # embedding serie anomale
        print("Embedding MoCo: serie anomale", flush=True)
        q_anom, k_anom = embed(anom_ids)
        print(f"OK -> anomale: {len(q_anom)} serie, q {q_anom.shape[1]} + k {k_anom.shape[1]} = {q_anom.shape[1] + k_anom.shape[1]} dim", "\n", flush=True)
 
    # concatenazione
    w_fit = np.concatenate([q_fit, k_fit], axis=1).astype(np.float64)
    w_thr = np.concatenate([q_thr, k_thr], axis=1).astype(np.float64)
    w_normal = np.concatenate([q_normal, k_normal], axis=1).astype(np.float64)
    w_anom = np.concatenate([q_anom, k_anom],axis=1).astype(np.float64)
 
    if standardize:
        scaler = StandardScaler().fit(w_fit)
        w_fit, w_thr, w_normal, w_anom = scaler.transform(w_fit), scaler.transform(w_thr), scaler.transform(w_normal), scaler.transform(w_anom)

    pw = PCA(whiten=True).fit(w_fit)
    z_fit = pw.transform(w_fit)
    z_thr = pw.transform(w_thr)
    z_normal = pw.transform(w_normal)
    z_anom = pw.transform(w_anom)    
 
    gamma = ocsvm_gamma if ocsvm_gamma in ("scale", "auto") else float(ocsvm_gamma)
    print(f"nu={ocsvm_nu}, gamma={gamma}, standardize={standardize})", flush=True)
 
    ocsvm = OneClassSVM(kernel="rbf", nu=ocsvm_nu, gamma=gamma).fit(z_fit)
 
    # punteggi: decision_function > 0 dentro il confine , < 0 fuori
    score_fit = ocsvm.decision_function(z_fit).astype(np.float64)
    score_thr = ocsvm.decision_function(z_thr).astype(np.float64)
    score_normal = ocsvm.decision_function(z_normal).astype(np.float64)
    score_anom = ocsvm.decision_function(z_anom).astype(np.float64)
 
    y = np.r_[np.zeros(len(score_normal)), np.ones(len(score_anom))]
    auroc = roc_auc_score(y, -np.r_[score_normal, score_anom])  

    frac_outside_fit = float((score_fit < 0).mean())
    print(f"auroc: {auroc:.4f} | fuori dal confine (fit): {frac_outside_fit:.3f}", flush=True)
 
    # salvataggio e risultati
    out_dir = f"/data/anomaly_results_ocsvm_{job_name}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
 
    # punteggi
    df = pd.DataFrame({
        "ID": normal_test_ids + anom_ids,
        "label": [0] * len(score_normal) + [1] * len(score_anom),
        "score": np.concatenate([score_normal, score_anom]),
    })
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    pd.DataFrame({"ID": normal_thr_ids, "score": score_thr}).to_csv(os.path.join(out_dir, "scores_thr.csv"), index=False)   # nuovo

 
    # soglia
    false_allarms = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)                             
    thr_scores = score_thr        
    eval_norm = score_normal                          
    anom = df[df.label == 1].score.values
    
    rules = {}
    for fa in false_allarms:
        thr = float(np.quantile(thr_scores, fa))
        tp = int((anom < thr).sum()); fn = len(anom) - tp
        fp = int((eval_norm < thr).sum()); tn = len(eval_norm) - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)

        rules[f"FA {fa:.0%}"] = {
            "soglia": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp), "REC": rec, "PREC": prec,
            "F1": 2 * prec * rec / (prec + rec) if tp > 0 else 0.0,
        }
        print(f"FA {fa:.0%}: soglia {thr:.4f} | TP {tp} FP {fp} FN {fn} TN {tn} | FA {fp/(fp+tn):.3f} REC {rec:.3f} PREC {prec:.3f}", flush=True)
 
    # vettori 
    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        fit_ids=np.array(normal_fit_ids), fit_q=q_fit, fit_k=k_fit,
        thr_ids=np.array(normal_thr_ids), thr_q=q_thr, thr_k=k_thr,                                 
        normal_ids=np.array(normal_test_ids), normal_q=q_normal, normal_k=k_normal,
        anom_ids=np.array(anom_ids), anom_q=q_anom, anom_k=k_anom,
    )
 
    # riassunto
    results = {
        "job_name": job_name, "metodo": "ocsvm", "dataset_anomalies": dataset_anomalies,
        "moco_version": moco_version, "min_flood_area": min_flood_area, "standardize": standardize,
        "ocsvm": {"nu": ocsvm_nu, "gamma": gamma, "dim": int(z_fit.shape[1]), "frac_outside_fit": frac_outside_fit},
        "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),   
        "auroc": float(auroc),
        "rules": rules,
    }
    with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
        json.dump(results, f, indent=2)
 
    shutil.make_archive(out_dir, "zip", out_dir)
    project_work.log_artifact(name=f"anomaly_results_ocsvm_{job_name}", source=f"{out_dir}.zip", kind="artifact")
 
    return "TERMINATO -> anomaly detection ocsvm"








@handler()
def ad_pca(
    job_name: str = "nome_job",
    old_vectors: str = "",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
    attn: bool = False,
    hidden_channels_dim: int = 64,
    pool_grid: int = 1,
    moco_dim: int = 128,
    moco_k: int = 4096,
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    standardize: bool = True,                       
    batch_size: int = 32,
    workers: int = 0,
    min_flood_area: float = 10.0,                    
    k_pca: int = 8,
    n_series_thr: int = 500,
    moco_attn: bool = False,
):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
  
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    # lettura ids serie anomale
    print(f"Lettura serie anomale: {dataset_anomalies}", flush=True)
    anom_path = project_data.get_artifact(f"{dataset_anomalies}").download(f"/data/{dataset_anomalies}")
    ids_per_sensor_anom = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(anom_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_anom[sensor] = ids
 
    anom_ids = ids_per_sensor_anom["SAR"] & ids_per_sensor_anom["OPT"]
    num_fit_series = len(anom_ids)
 
    # filtro superficie alluvione
    if min_flood_area > 0:
        try:
            mask_parts = sorted(glob(os.path.join(anom_path, "MASK", "part*.zip")))

            flood_area = {}
            for part_path in mask_parts:
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('.npy') and name[:-4] in anom_ids:
                            mask = np.load(io.BytesIO(z.read(name)))
                            flood_area[name[:-4]] = 100.0 * float((mask == 2).sum()) / mask.size

            anom_ids = {ID for ID in anom_ids if flood_area.get(ID, 0.0) >= min_flood_area}
            print(f"OK -> flood >= {min_flood_area:g}%: {len(anom_ids)} serie su {len(anom_ids)}", flush=True)

        except Exception as e:
            print(f"EXC -> flood_ {e}", flush=True)
 
    anom_ids = sorted(anom_ids)
    print(f"OK -> {len(anom_ids)} serie anomale", "\n", flush=True)


    # lettura serie normali
    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(f"/data/Floods_{dataset_standard}_crop_norm")

    # import ids lits from moco training
    if moco_attn:
        p = project_work.get_artifact(f"moco_attn_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
    else:
        p = project_work.get_artifact(f"moco_2D_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
    
    with open(p, 'r') as f:
        test_ids = json.load(f)
    print(f"OK -> liste normali caricate: {len(test_ids)} test", flush=True)
    num_fit_series = len(test_ids) - num_fit_series
 
    random.seed(0)
    shuffled_test_ids = random.sample(test_ids, len(test_ids)) 

    ids_per_sensor_normal = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(normal_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_normal[sensor] = ids
 
    normal_ids = ids_per_sensor_normal["SAR"] & ids_per_sensor_normal["OPT"]

    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID in normal_ids]

    normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]                            # fit -> k_pca
    normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]              # calcolo soglie
    normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]             # anomaly detection 


    print(f"OK -> {len(test_ids)} serie normali tot, {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali")

    if old_vectors:
        print(f"Upload embedding: anomaly_results_pca_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_pca_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            z.extract("embeddings.npz", "/data/old_vectors")
            old_results = json.loads(z.read("anomaly_results.json"))

        for key, value in (("moco_version", moco_version), ("dataset_anomalies", dataset_anomalies)):
            if old_results.get(key) != value:
                raise ValueError(f"old_vectors={old_vectors} ha {key}={old_results.get(key)}, questo job ha {value}")

        old = np.load("/data/old_vectors/embeddings.npz")
        lookup = {}
        for g in ("fit", "thr", "normal", "anom"):
            for i, ID in enumerate(old[f"{g}_ids"]):
                lookup[str(ID)] = (old[f"{g}_q"][i], old[f"{g}_k"][i])

        def take(ids, nome):
            missing = [ID for ID in ids if ID not in lookup]
            if missing:
                raise ValueError(f"{len(missing)} serie {nome} non presenti in {old_vectors} (es. {missing[:3]})")
            return np.stack([lookup[ID][0] for ID in ids]), np.stack([lookup[ID][1] for ID in ids])

        q_fit, k_fit = take(normal_fit_ids, "fit")
        q_thr, k_thr = take(normal_thr_ids, "thr")
        q_normal, k_normal = take(normal_test_ids, "normali test")
        q_anom, k_anom = take(anom_ids, "anomale")
        print(f"OK -> embedding caricatu: {len(q_fit)} fit,  {len(q_thr)} thr, {len(q_normal)} normali, {len(q_anom)} anomale", "\n", flush=True)

    else:

        print("calcolo vettori MoCo", flush = True)

        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"

        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids)}       
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)

        # caricamento pesi moco
        try:
            if moco_attn:
                encoderSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)
                encoderOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)

                model = MoCo2encoders_attn(
                    base_encoder_q=encoderOPT.encoder,
                    base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    symmetric=False, 
                    hidden_channels_dim=hidden_channels_dim,
                    pool_grid=pool_grid,
                    simple_proj=False,
                    attn=attn
                ).to(device)

                w_path = project_work.get_artifact(f"moco_attn_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)

            else:
                encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
                encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)

                model = MoCo2encoders_2d(
                    base_encoder_q=encoderOPT.encoder,
                    base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    symmetric=False, device=device,
                    hidden_channels_dim=hidden_channels_dim,
                    simple_proj=simple_proj,
                    pool_grid=pool_grid,
                    attn=attn
                ).to(device)

                w_path = project_work.get_artifact(f"moco_2D_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)               
            
            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("OK -> modello MoCo caricato",  "\n", flush=True)

        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)

        # embedding moco: pesi congelati
        # 2 matrici: rows = serie, cols = dim
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            out_q, out_k = [], []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)
                    _, q, k = model.contrastive_loss(im_q, im_k)
                    out_q.append(q.float().flatten(1).cpu())
                    out_k.append(k.float().flatten(1).cpu())
            return torch.cat(out_q).numpy(), torch.cat(out_k).numpy()

        # embedding serie di fit 
        print("Embedding MoCo: serie di fit pac", flush=True)
        q_fit, k_fit = embed(normal_fit_ids)
        print(f"OK -> normali fit: {len(q_fit)} serie, q {q_fit.shape[1]} + k {k_fit.shape[1]} = {q_fit.shape[1] + k_fit.shape[1]} dim", "\n", flush=True)

        # embedding serie thr 
        print("Embedding MoCo: serie thr pac", flush=True)
        q_thr, k_thr = embed(normal_thr_ids)
        print(f"OK -> normali thr: {len(q_thr)} serie, q {q_thr.shape[1]} + k {k_thr.shape[1]} = {q_thr.shape[1] + k_thr.shape[1]} dim", "\n", flush=True)
 
        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        q_normal, k_normal = embed(normal_test_ids)
        print(f"OK -> normali test: {len(q_normal)} serie, q {q_normal.shape[1]} + k {k_normal.shape[1]} = {q_normal.shape[1] + k_normal.shape[1]} dim", "\n", flush=True)

        # pulizia cartelle
        for d in ("/data/cache_SAR", "/data/cache_OPT"):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        # estrazione serie anomale
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_anom_series = {f"{ID}.npy" for ID in anom_ids}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(anom_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_anom_series:
                            z.extract(name, cache_dir)

        # embedding serie anomale
        print("Embedding MoCo: serie anomale", flush=True)
        q_anom, k_anom = embed(anom_ids)
        print(f"OK -> anomale: {len(q_anom)} serie, q {q_anom.shape[1]} + k {k_anom.shape[1]} = {q_anom.shape[1] + k_anom.shape[1]} dim", "\n", flush=True)
 
    # concatenazione
    z_fit = np.concatenate([q_fit, k_fit], axis=1)
    z_thr = np.concatenate([q_thr, k_thr], axis=1)
    z_normal = np.concatenate([q_normal, k_normal], axis=1)
    z_anom = np.concatenate([q_anom, k_anom],axis=1)

    if standardize:
        scaler = StandardScaler().fit(z_fit)
        z_fit, z_thr, z_normal, z_anom = scaler.transform(z_fit), scaler.transform(z_thr), scaler.transform(z_normal), scaler.transform(z_anom)

    pca = PCA().fit(z_fit)                      
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(np.searchsorted(cum, [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.99]) + 1)     
    # [ 1  2  5  9 13 19 25 43 55 63 73]

    pca = PCA(n_components=k_pca).fit(z_fit)
    res_thr = ((z_thr - pca.inverse_transform(pca.transform(z_thr))) ** 2).sum(axis=1)             
    res_normal = ((z_normal - pca.inverse_transform(pca.transform(z_normal))) ** 2).sum(axis=1)
    res_anom = ((z_anom - pca.inverse_transform(pca.transform(z_anom))) ** 2).sum(axis=1)

    y = np.r_[np.zeros(len(res_normal)), np.ones(len(res_anom))]
    print(roc_auc_score(y, np.r_[res_normal, res_anom]))

    # salvataggio e risultati
    out_dir = f"/data/anomaly_results_pca_{job_name}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    
    # punteggi
    score_normal = -res_normal.astype(np.float64)
    score_anom = -res_anom.astype(np.float64)
    score_thr = -res_thr.astype(np.float64) 
    auroc = roc_auc_score(np.r_[np.zeros(len(res_normal)), np.ones(len(res_anom))], np.r_[res_normal, res_anom])
    
    df = pd.DataFrame({
        "ID": normal_test_ids + anom_ids,
        "label": [0] * len(score_normal) + [1] * len(score_anom),
        "score": np.concatenate([score_normal, score_anom]),
    })
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    pd.DataFrame({"ID": normal_thr_ids, "score": score_thr}).to_csv(os.path.join(out_dir, "scores_thr.csv"), index=False)   # nuovo

    
    # soglia
    false_allarms = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)                             
    thr_scores = score_thr        
    eval_norm = score_normal                          
    anom = df[df.label == 1].score.values
    
    rules = {}
    for fa in false_allarms:
        thr = float(np.quantile(thr_scores, fa))
        tp = int((anom < thr).sum()); fn = len(anom) - tp
        fp = int((eval_norm < thr).sum()); tn = len(eval_norm) - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)

        rules[f"score < FA {fa:.0%})"] = {
            "soglia": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp), "REC": rec, "PREC": prec,
            "F1": 2 * prec * rec / (prec + rec) if tp > 0 else 0.0,
        }
        print(f"FA {fa:.0%}: soglia {thr:.4f} | TP {tp} FP {fp} FN {fn} TN {tn} | FA {fp/(fp+tn):.3f} REC {rec:.3f} PREC {prec:.3f}", flush=True)

    # vettori 
    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        fit_ids=np.array(normal_fit_ids), fit_q=q_fit, fit_k=k_fit,
        thr_ids=np.array(normal_thr_ids), thr_q=q_thr, thr_k=k_thr,                                 
        normal_ids=np.array(normal_test_ids), normal_q=q_normal, normal_k=k_normal,
        anom_ids=np.array(anom_ids), anom_q=q_anom, anom_k=k_anom,
    )
    
    # riassunto
    results = {
        "job_name": job_name, "metodo": "pca_residuo", "dataset_anomalies": dataset_anomalies,
        "moco_version": moco_version, "min_flood_area": min_flood_area, "standardize": standardize,
        "k": int(k_pca), "varianza": float(cum[k_pca - 1]),
        "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),   
        "auroc": float(auroc),
        "rules": rules,
    }
    with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    shutil.make_archive(out_dir, "zip", out_dir)
    project_work.log_artifact(name=f"anomaly_results_pca_{job_name}", source=f"{out_dir}.zip", kind="artifact")




@handler()
def ad_md(
    job_name: str = "nome_job",
    old_vectors: str = "",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
    attn: bool = False,
    hidden_channels_dim: int = 64,
    pool_grid: int = 1,
    moco_dim: int = 128,
    moco_k: int = 4096,
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    standardize: bool = True,                       
    batch_size: int = 32,
    workers: int = 0,
    min_flood_area: float = 10.0,
    n_series_thr: int = 500,   
    moco_attn: bool = False,                 
):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
  
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    # lettura ids serie anomale
    print(f"Lettura serie anomale: {dataset_anomalies}", flush=True)
    anom_path = project_data.get_artifact(f"{dataset_anomalies}").download(f"/data/{dataset_anomalies}")
    ids_per_sensor_anom = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(anom_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_anom[sensor] = ids
 
    anom_ids = ids_per_sensor_anom["SAR"] & ids_per_sensor_anom["OPT"]
    num_fit_series = len(anom_ids)
 
    # filtro superficie alluvione
    if min_flood_area > 0:
        try:
            mask_parts = sorted(glob(os.path.join(anom_path, "MASK", "part*.zip")))

            flood_area = {}
            for part_path in mask_parts:
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('.npy') and name[:-4] in anom_ids:
                            mask = np.load(io.BytesIO(z.read(name)))
                            flood_area[name[:-4]] = 100.0 * float((mask == 2).sum()) / mask.size

            anom_ids = {ID for ID in anom_ids if flood_area.get(ID, 0.0) >= min_flood_area}
            print(f"OK -> flood >= {min_flood_area:g}%: {len(anom_ids)} serie su {len(anom_ids)}", flush=True)

        except Exception as e:
            print(f"EXC -> flood: {e}", flush=True)
 
    anom_ids = sorted(anom_ids)
    print(f"OK -> {len(anom_ids)} serie anomale", "\n", flush=True)


    # lettura serie normali
    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(f"/data/Floods_{dataset_standard}_crop_norm")

    # import ids lits from moco training
    if moco_attn:
        p = project_work.get_artifact(f"moco_attn_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
    else:
        p = project_work.get_artifact(f"moco_2D_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)

    with open(p, 'r') as f:
        test_ids = json.load(f)
    print(f"OK -> liste normali caricate: {len(test_ids)} test", flush=True)
    num_fit_series = len(test_ids) - num_fit_series
 
    random.seed(0)
    shuffled_test_ids = random.sample(test_ids, len(test_ids)) 

    ids_per_sensor_normal = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(normal_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_normal[sensor] = ids
 
    normal_ids = ids_per_sensor_normal["SAR"] & ids_per_sensor_normal["OPT"]
 
    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID in normal_ids]

    normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]                            # fit -> var
    normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]              # calcolo soglie
    normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]             # anomaly detection 


    print(f"OK -> {len(test_ids)} serie normali tot, {len(normal_fit_ids)} serie fit, {len(normal_test_ids)} serie test normali")

    if old_vectors:
        print(f"Upload embedding: anomaly_results_md_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_md_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            z.extract("embeddings.npz", "/data/old_vectors")
            old_results = json.loads(z.read("anomaly_results.json"))

        for key, value in (("moco_version", moco_version), ("dataset_anomalies", dataset_anomalies)):
            if old_results.get(key) != value:
                raise ValueError(f"old_vectors={old_vectors} ha {key}={old_results.get(key)}, questo job ha {value}")

        old = np.load("/data/old_vectors/embeddings.npz")
        lookup = {}
        for g in ("fit", "thr", "normal", "anom"):
            for i, ID in enumerate(old[f"{g}_ids"]):
                lookup[str(ID)] = (old[f"{g}_q"][i], old[f"{g}_k"][i])

        def take(ids, nome):
            missing = [ID for ID in ids if ID not in lookup]
            if missing:
                raise ValueError(f"{len(missing)} serie {nome} non presenti in {old_vectors} (es. {missing[:3]})")
            return np.stack([lookup[ID][0] for ID in ids]), np.stack([lookup[ID][1] for ID in ids])

        q_fit, k_fit = take(normal_fit_ids, "fit")
        q_thr, k_thr = take(normal_thr_ids, "thr")
        q_normal, k_normal = take(normal_test_ids, "normali test")
        q_anom, k_anom = take(anom_ids, "anomale")
        print(f"OK -> embedding caricatu: {len(q_fit)} fit,  {len(q_thr)} thr, {len(q_normal)} normali, {len(q_anom)} anomale", "\n", flush=True)

    else:

        print("calcolo vettori MoCo", flush = True)

        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"

        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids)}       
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)

        # caricamento pesi moco
        try:
            if moco_attn:
                encoderSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)
                encoderOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)

                model = MoCo2encoders_attn(
                    base_encoder_q=encoderOPT.encoder,
                    base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    symmetric=False, 
                    hidden_channels_dim=hidden_channels_dim,
                    pool_grid=pool_grid,
                    simple_proj=False,
                    attn=attn
                ).to(device)

                w_path = project_work.get_artifact(f"moco_attn_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)

            else:
                encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
                encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)

                model = MoCo2encoders_2d(
                    base_encoder_q=encoderOPT.encoder,
                    base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    symmetric=False, device=device,
                    hidden_channels_dim=hidden_channels_dim,
                    simple_proj=simple_proj,
                    pool_grid=pool_grid,
                    attn=attn
                ).to(device)

                w_path = project_work.get_artifact(f"moco_2D_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)               
            
            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("OK -> modello MoCo caricato",  "\n", flush=True)

        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)

        # embedding moco: pesi congelati
        # 2 matrici: rows = serie, cols = dim
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            out_q, out_k = [], []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)
                    _, q, k = model.contrastive_loss(im_q, im_k)
                    out_q.append(q.float().flatten(1).cpu())
                    out_k.append(k.float().flatten(1).cpu())
            return torch.cat(out_q).numpy(), torch.cat(out_k).numpy()

        # embedding serie di fit 
        print("Embedding MoCo: serie di fit pac", flush=True)
        q_fit, k_fit = embed(normal_fit_ids)
        print(f"OK -> normali fit: {len(q_fit)} serie, q {q_fit.shape[1]} + k {k_fit.shape[1]} = {q_fit.shape[1] + k_fit.shape[1]} dim", "\n", flush=True)

        # embedding serie thr 
        print("Embedding MoCo: serie thr pac", flush=True)
        q_thr, k_thr = embed(normal_thr_ids)
        print(f"OK -> normali thr: {len(q_thr)} serie, q {q_thr.shape[1]} + k {k_thr.shape[1]} = {q_thr.shape[1] + k_thr.shape[1]} dim", "\n", flush=True)
 
        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        q_normal, k_normal = embed(normal_test_ids)
        print(f"OK -> normali test: {len(q_normal)} serie, q {q_normal.shape[1]} + k {k_normal.shape[1]} = {q_normal.shape[1] + k_normal.shape[1]} dim", "\n", flush=True)

        # pulizia cartelle
        for d in ("/data/cache_SAR", "/data/cache_OPT"):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        # estrazione serie anomale
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_anom_series = {f"{ID}.npy" for ID in anom_ids}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(anom_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_anom_series:
                            z.extract(name, cache_dir)

        # embedding serie anomale
        print("Embedding MoCo: serie anomale", flush=True)
        q_anom, k_anom = embed(anom_ids)
        print(f"OK -> anomale: {len(q_anom)} serie, q {q_anom.shape[1]} + k {k_anom.shape[1]} = {q_anom.shape[1] + k_anom.shape[1]} dim", "\n", flush=True)
 
    # concatenazione
    z_fit = np.concatenate([q_fit, k_fit], axis=1)
    z_thr = np.concatenate([q_thr, k_thr], axis=1)
    z_normal = np.concatenate([q_normal, k_normal], axis=1)
    z_anom = np.concatenate([q_anom, k_anom],axis=1)

    if standardize:
        scaler = StandardScaler().fit(z_fit)
        z_fit, z_thr, z_normal, z_anom = scaler.transform(z_fit), scaler.transform(z_thr), scaler.transform(z_normal), scaler.transform(z_anom)

    # distanza mahalanobis
    pca = PCA(whiten=True).fit(z_fit.astype(np.float64))
    res_thr = (pca.transform(z_thr.astype(np.float64)) ** 2).sum(axis=1)
    res_normal = (pca.transform(z_normal.astype(np.float64)) ** 2).sum(axis=1)
    res_anom = (pca.transform(z_anom.astype(np.float64)) ** 2).sum(axis=1)

    y = np.r_[np.zeros(len(res_normal)), np.ones(len(res_anom))]
    print(roc_auc_score(y, np.r_[res_normal, res_anom]))

    # salvataggio e risultati
    out_dir = f"/data/anomaly_results_md_{job_name}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    
    # punteggi
    score_normal = -res_normal.astype(np.float64)
    score_anom = -res_anom.astype(np.float64)
    score_thr = -res_thr.astype(np.float64)
    auroc = roc_auc_score(np.r_[np.zeros(len(res_normal)), np.ones(len(res_anom))], np.r_[res_normal, res_anom])
    
    df = pd.DataFrame({
        "ID": normal_test_ids + anom_ids,
        "label": [0] * len(score_normal) + [1] * len(score_anom),
        "score": np.concatenate([score_normal, score_anom]),
    })
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    
    # soglia
    false_allarms = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)                             
    thr_scores = score_thr        
    eval_norm = score_normal                          
    anom = df[df.label == 1].score.values
    
    rules = {}
    for fa in false_allarms:
        thr = float(np.quantile(thr_scores, fa))
        tp = int((anom < thr).sum()); fn = len(anom) - tp
        fp = int((eval_norm < thr).sum()); tn = len(eval_norm) - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)

        rules[f"FA {fa:.0%}"] = {
            "soglia": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp), "REC": rec, "PREC": prec,
            "F1": 2 * prec * rec / (prec + rec) if tp > 0 else 0.0,
        }
        print(f"FA {fa:.0%}: soglia {thr:.4f} | TP {tp} FP {fp} FN {fn} TN {tn} | FA {fp/(fp+tn):.3f} REC {rec:.3f} PREC {prec:.3f}", flush=True)
    
    # vettori 
    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        fit_ids=np.array(normal_fit_ids), fit_q=q_fit, fit_k=k_fit,
        thr_ids=np.array(normal_thr_ids), thr_q=q_thr, thr_k=k_thr,                                 
        normal_ids=np.array(normal_test_ids), normal_q=q_normal, normal_k=k_normal,
        anom_ids=np.array(anom_ids), anom_q=q_anom, anom_k=k_anom,
    )
    
    # riassunto
    results = {
        "job_name": job_name, "metodo": "md", "dataset_anomalies": dataset_anomalies,
        "moco_version": moco_version, "min_flood_area": min_flood_area, "standardize": standardize,
        "whitening": True, "n_componenti": int(pca.n_components_),
        "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
        "auroc": float(auroc),
        "rules": rules,
    }
    with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    shutil.make_archive(out_dir, "zip", out_dir)
    project_work.log_artifact(name=f"anomaly_results_md_{job_name}", source=f"{out_dir}.zip", kind="artifact")











@handler()
def linear_regression(
    job_name: str = "linReg_v1",
    versions: list | None = None,                   # versioni
    dataset_standard: str = "Standard",
    include_normals: bool = True,
    n_folds: int = 5,
    seed: int = 0,
    batch_size: int = 16,
    workers: int = 0,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    hidden_channels_dim: int = 64,
    moco_dim: int = 128, moco_k: int = 4096, moco_m: float = 0.999, moco_t: float = 0.07,
    save_features: bool = False,                    
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "\n", flush=True)

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    if not versions:
        raise ValueError("versions vuoto")

    work_dir = "/data/linear"
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)

    # calcolo id
    ids_anom, ids_normal = [], []
    dataset_anomalies = None

    for v in versions:
        art = v["results_artifact"]
        p = project_work.get_artifact(art).download(os.path.join(work_dir, f"{art}.zip"), overwrite=True)
        with zipfile.ZipFile(p) as z:
            res = json.loads(z.read("anomaly_results.json"))
            z.extract("embeddings.npz", os.path.join(work_dir, art))
        emb = np.load(os.path.join(work_dir, art, "embeddings.npz"))

        if dataset_anomalies is None:
            dataset_anomalies = res["dataset_anomalies"]

        ids_anom.append({str(i) for i in emb["anom_ids"]})
        ids_normal.append({str(i) for i in emb["normal_ids"]})

    anom_ids = sorted(set.intersection(*ids_anom))
    normal_ids = sorted(set.intersection(*ids_normal)) if include_normals else []
    print(f"OK -> serie comuni: {len(anom_ids)} anomale, {len(normal_ids)} normali", "\n", flush=True)

    # calcolo percentuale alluvione
    print(f"Download anomale: {dataset_anomalies}", flush=True)
    anom_path = project_data.get_artifact(dataset_anomalies).download(f"/data/{dataset_anomalies}", overwrite=True)

    wanted = set(anom_ids)
    flood_area = {}
    for part_path in sorted(glob(os.path.join(anom_path, "MASK", "part*.zip"))):
        with zipfile.ZipFile(part_path, "r") as z:
            for name in z.namelist():
                if name.endswith(".npy") and name[:-4] in wanted:
                    mask = np.load(io.BytesIO(z.read(name)))
                    flood_area[name[:-4]] = 100.0 * float((mask == 2).sum()) / mask.size

    missing = [ID for ID in anom_ids if ID not in flood_area]
    if missing:
        print(f"ATT -> {len(missing)} serie senza maschera", flush=True)
        anom_ids = [ID for ID in anom_ids if ID in flood_area]

    y_anom = np.array([flood_area[ID] for ID in anom_ids])
    print(f"OK -> area allagata: media {y_anom.mean():.2f}% | mediana {np.median(y_anom):.2f}%", "\n", flush=True)

    normal_path = None
    if normal_ids:
        print(f"Download normali: Floods_{dataset_standard}_crop_norm", flush=True)
        normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(
            f"/data/Floods_{dataset_standard}_crop_norm", overwrite=True)

    # modello

    def build_model(v):
        if v["moco_attn"]:
            print("moco con attn", flush=True)
            encSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=v.get("ltae", True), spac_net=v.get("spac_net", False)).to(device)
            encOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=v.get("ltae", True), spac_net=v.get("spac_net", False)).to(device)

            model = MoCo2encoders_attn(
                base_encoder_q=encOPT.encoder, 
                base_encoder_k=encSAR.encoder,
                dim=moco_dim, 
                K=moco_k, 
                m=moco_m, 
                T=moco_t, 
                symmetric=False,
                hidden_channels_dim=hidden_channels_dim, 
                pool_grid=v["pool_grid"],
                simple_proj=False, 
                attn=False,
            ).to(device)

            w_name = f"moco_attn_weights_{v['moco_version']}"

        else:
            print("moco senza attn, 2D", flush=True)
            encSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
            encOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)
            
            model = MoCo2encoders_2d(
                base_encoder_q=encOPT.encoder, 
                base_encoder_k=encSAR.encoder,
                dim=moco_dim, 
                K=moco_k, 
                m=moco_m, 
                T=moco_t, 
                symmetric=False, 
                device=device,
                hidden_channels_dim=hidden_channels_dim, 
                simple_proj=v.get("simple_proj", False),
                pool_grid=v["pool_grid"], attn=v["attn"],
            ).to(device)

            w_name = f"moco_2D_weights_{v['moco_version']}"

        w_path = project_work.get_artifact(w_name).download(os.path.join(work_dir, f"{w_name}.pth"), overwrite=True)
        check = torch.load(w_path, map_location=device)
        state_dict = check["model"] if isinstance(check, dict) and "model" in check else check
        model.load_state_dict(state_dict)      
        model.eval()
        print(f"OK -> modello caricato: {w_name}", flush=True)
        return model

    # estrazione 3 punti
    def branch_features(model, x, ramo):
        enc  = model.encoder_q if ramo == "q" else model.encoder_k
        pool = model.pool_q    if ramo == "q" else model.pool_k
        proj = model.proj_q    if ramo == "q" else model.proj_k

        z = enc(x)                                              

        # autoencoder -> dim(B, 256)
        f_enc = F.adaptive_avg_pool2d(z, 4).flatten(1)          

        # moco 
        if model.simple_proj:
            h = pool(z).flatten(1)
        else:
            # moco 1 dopo conv 
            conv = model.conv_q if ramo == "q" else model.conv_k
            h = conv(z)                                        
            h = pool(h).flatten(1)
        f_pre = h                                               

        # moco 2 dopo proj -> (B, moco_dim)
        f_post = F.normalize(proj(h), dim=1)                    
        return f_enc, f_pre, f_post


    # estrazione serie
    def extract(model, ids):
        dataset = MoCo2encodersLoader(
            listIDs=ids, sar_map={}, opt_map={}, transform=None, patch_size=patch_size,
            n_images1=n_images1, n_channels1=n_channels1, n_images2=n_images2, n_channels2=n_channels2,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

        out = {(p, m): [] for p in ("encoder", "pre_proj", "post_proj") for m in ("OTTICO", "SAR")}
        with torch.no_grad(), torch.cuda.amp.autocast():
            for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = ottico, im_k = sar
                for ramo, mod, x in (("q", "OTTICO", im_q), ("k", "SAR", im_k)):
                    for p, f in zip(("encoder", "pre_proj", "post_proj"), branch_features(model, x.to(device), ramo)):
                        out[(p, mod)].append(f.float().cpu())
        return {key: torch.cat(val).numpy() for key, val in out.items()}

    def fill_cache(src_path, ids):
        for d in ("/data/cache_SAR", "/data/cache_OPT"):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        wanted = {f"{ID}.npy" for ID in ids}
        for modality, cache_dir in (("SAR", "/data/cache_SAR"), ("OPT", "/data/cache_OPT")):
            for part_path in sorted(glob(os.path.join(src_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, "r") as z:
                    for name in z.namelist():
                        if name in wanted:
                            z.extract(name, cache_dir)

    feats = {v["name"]: {} for v in versions}       
    groups = [("anom", anom_path, anom_ids)] + ([("normal", normal_path, normal_ids)] if normal_ids else [])

    for gruppo, src_path, ids in groups:
        print(f"Estrazione serie {gruppo}: {len(ids)}", flush=True)
        fill_cache(src_path, ids)
        for v in versions:
            model = build_model(v)
            feats[v["name"]][gruppo] = extract(model, ids)
            del model
            torch.cuda.empty_cache()
        print(flush=True)

    # linear regression
    bins = [(0, 0), (0, 5), (5, 20), (20, 100), (0, 100)]

    def linReg(X, y):
        cv = KFold(n_folds, shuffle=True, random_state=seed)          
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        pred = cross_val_predict(model, X, y, cv=cv)

        out = {
            "R2": 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "MAE": float(np.abs(y - pred).mean()),
            "Spearman": float(spearmanr(y, pred).correlation),
        }
        norm = (y == 0)
        for lo, hi in bins:
            sel = norm if hi == 0 else ((y > lo) & (y <= hi))
            nome = "0%" if hi == 0 else f"{lo}-{hi}%"
            out[f"MAE_{nome}"] = float(np.abs(y[sel] - pred[sel]).mean()) if sel.any() else np.nan
            out[f"BIAS_{nome}"] = float((pred[sel] - y[sel]).mean()) if sel.any() else np.nan
            if hi == 0:
                continue
            out[f"Spearman_{nome}"] = float(spearmanr(y[sel], pred[sel]).correlation) if sel.sum() > 2 else np.nan
            if norm.any() and sel.any():
                etichette = np.r_[np.zeros(norm.sum()), np.ones(sel.sum())]
                out[f"AUROC_0vs{nome}"] = float(roc_auc_score(etichette, np.r_[pred[norm], pred[sel]]))
            else:
                out[f"AUROC_0vs{nome}"] = np.nan
        return out

    settings = [("anomale", False)] + ([("anomale+normali", True)] if normal_ids else [])
    rows = []

    for v in versions:
        f_v = feats[v["name"]]
        for punto in ("encoder", "pre_proj", "post_proj"):
            for setting, with_normals in settings:
                y = np.r_[y_anom, np.zeros(len(normal_ids))] if with_normals else y_anom

                def get(mod):
                    a = f_v["anom"][(punto, mod)]
                    return np.concatenate([a, f_v["normal"][(punto, mod)]]) if with_normals else a

                q, k = get("OTTICO"), get("SAR")
                for modalita, X in (("SAR", k), ("OTTICO", q), ("SAR+OTTICO", np.concatenate([q, k], axis=1))):
                    m = linReg(X, y)
                    rows.append({"versione": v["name"], "punto": punto, "serie": setting, "modalita": modalita,
                                 "n": len(y), "dim": X.shape[1], **m})
                    print(f"{v['name']} | {punto:9s} | {setting:15s} | {modalita:10s} | "
                          f"R² {m['R2']:.3f} | MAE {m['MAE']:.2f}% | Spearman {m['Spearman']:.3f}", flush=True)
        print(flush=True)

    # salvataggio risultati
    out_dir = f"/data/linear_probe_{job_name}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "linear_probe.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"job_name": job_name, "versions": versions, "dataset_anomalies": dataset_anomalies,
                   "n_anomale": len(anom_ids), "n_normali": len(normal_ids),
                   "n_folds": n_folds, "seed": seed}, f, indent=2)

    if save_features:
        for v in versions:
            arrays = {"anom_ids": np.array(anom_ids), "normal_ids": np.array(normal_ids), "y_anom": y_anom}
            for gruppo, d in feats[v["name"]].items():
                for (punto, mod), arr in d.items():
                    arrays[f"{gruppo}_{punto}_{mod}"] = arr
            np.savez_compressed(os.path.join(out_dir, f"features_{v['name']}.npz"), **arrays)

    shutil.make_archive(out_dir, "zip", out_dir)
    try:
        project_work.log_artifact(name=f"linear_probe_{job_name}", source=f"{out_dir}.zip", kind="artifact")
        print(f"OK -> risultati salvati: linear_probe_{job_name}", flush=True)
    except Exception as e:
        print(f"EXC -> upload risultati: {e}", flush=True)

    return "TERMINATO -> linear regression"






# + import delle tue classi: Singlemodal_CAE_attn, Singlemodal_CAE_2d,
#   MoCo2encoders_attn, MoCo2encoders_2d, MoCo2encodersLoader


@handler()
def ad_md_features(
    job_name: str = "nome_job",
    feature_point: str = "post_proj",               # "encoder" | "pre_proj" | "post_proj"
    pca_components: float | int | None = None,      # None = tutte; 0.99 = 99% varianza; 256 = numero fisso
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "Floods_Anomalies_v3_crop_norm",
    moco_version: str = "v1",
    moco_attn: bool = False,                        # True -> MoCo2encoders_attn, False -> MoCo2encoders_2d
    ltae: bool = True,                              # solo per moco_attn=True
    spac_net: bool = False,                         # solo per moco_attn=True
    simple_proj: bool = False,
    attn: bool = False,
    hidden_channels_dim: int = 64,
    pool_grid: int = 1,
    moco_dim: int = 128,
    moco_k: int = 4096,
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    standardize: bool = True,
    batch_size: int = 16,
    workers: int = 0,
    min_flood_area: float = 0.0,
    n_series_thr: int = 500,
    save_embeddings: bool = False,
):
    if feature_point not in ("encoder", "pre_proj", "proj_raw", "post_proj"):
        raise ValueError(f"feature_point non valido: {feature_point}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "\n", flush=True)

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    print(f"Lettura serie anomale: {dataset_anomalies}", flush=True)
    anom_path = project_data.get_artifact(dataset_anomalies).download(f"/data/{dataset_anomalies}", overwrite=True)

    ids_per_sensor = {}
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(anom_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, "r") as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith(".npy"))
        ids_per_sensor[sensor] = ids
    anom_ids = ids_per_sensor["SAR"] & ids_per_sensor["OPT"]
    n_anom_before_filter = len(anom_ids)            # usato per la suddivisione delle normali, come in ad_md

    flood_area = {}
    for part_path in sorted(glob(os.path.join(anom_path, "MASK", "part*.zip"))):
        with zipfile.ZipFile(part_path, "r") as z:
            for name in z.namelist():
                if name.endswith(".npy") and name[:-4] in anom_ids:
                    mask = np.load(io.BytesIO(z.read(name)))
                    flood_area[name[:-4]] = 100.0 * float((mask == 2).sum()) / mask.size

    anom_ids = sorted(ID for ID in anom_ids if flood_area.get(ID, 0.0) >= min_flood_area)
    print(f"OK -> flood >= {min_flood_area:g}%: {len(anom_ids)} serie su {n_anom_before_filter}", "\n", flush=True)


    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(
        f"/data/Floods_{dataset_standard}_crop_norm", overwrite=True)

    list_name = f"moco_attn_testIdsList_{moco_version}" if moco_attn else f"moco_2D_testIdsList_{moco_version}"
    p = project_work.get_artifact(list_name).download("/data/det_test_ids.json", overwrite=True)
    with open(p, "r") as f:
        test_ids = json.load(f)

    random.seed(0)
    shuffled_test_ids = random.sample(test_ids, len(test_ids))

    normal_available = None
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(normal_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, "r") as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith(".npy"))
        normal_available = ids if normal_available is None else normal_available & ids
    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID in normal_available]

    num_fit_series = len(test_ids) - n_anom_before_filter
    normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]
    normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]
    normal_test_ids = shuffled_test_ids[num_fit_series:num_fit_series + len(anom_ids)]

    assert len(normal_fit_ids) > 0 and len(normal_thr_ids) == n_series_thr, "serie normali insufficienti per fit/soglia"
    print(f"OK -> normali: {len(normal_fit_ids)} fit, {len(normal_thr_ids)} soglia, {len(normal_test_ids)} test", "\n", flush=True)


    if moco_attn:
        encSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=ltae, spac_net=spac_net).to(device)
        encOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=ltae, spac_net=spac_net).to(device)
        model = MoCo2encoders_attn(
            base_encoder_q=encOPT.encoder, base_encoder_k=encSAR.encoder,
            dim=moco_dim, K=moco_k, m=moco_m, T=moco_t, symmetric=False,
            hidden_channels_dim=hidden_channels_dim, pool_grid=pool_grid,
            simple_proj=simple_proj, attn=attn,
        ).to(device)
        w_name = f"moco_attn_weights_{moco_version}"
    else:
        encSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
        encOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)
        model = MoCo2encoders_2d(
            base_encoder_q=encOPT.encoder, base_encoder_k=encSAR.encoder,
            dim=moco_dim, K=moco_k, m=moco_m, T=moco_t, symmetric=False, device=device,
            hidden_channels_dim=hidden_channels_dim, simple_proj=simple_proj,
            pool_grid=pool_grid, 
        ).to(device)
        w_name = f"moco_2D_weights_{moco_version}"

    w_path = project_work.get_artifact(w_name).download("/data/moco_weights.pth", overwrite=True)
    ckpt = torch.load(w_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"OK -> modello caricato: {w_name}", "\n", flush=True)

    def branch_features(x, ramo):
        enc  = model.encoder_q if ramo == "q" else model.encoder_k
        pool = model.pool_q    if ramo == "q" else model.pool_k
        proj = model.proj_q    if ramo == "q" else model.proj_k

        z = enc(x)                                                  # (B, 16, 64, 64)
        if feature_point == "encoder":
            return F.adaptive_avg_pool2d(z, 4).flatten(1)           # (B, 256)

        if model.simple_proj:
            h = pool(z).flatten(1)
        else:
            conv = model.conv_q if ramo == "q" else model.conv_k
            h = conv(z)
            if getattr(model, "attn", False):
                attn_mod = model.attn_q if ramo == "q" else model.attn_k
                scores = attn_mod(h).flatten(2).softmax(dim=-1)
                h = torch.einsum("bcn,bkn->bck", h.flatten(2), scores).squeeze(-1)
            else:
                h = pool(h).flatten(1)
        if feature_point == "pre_proj":
            return h

        p = proj(h)                                   # uscita del proiettore
        if feature_point == "proj_raw":
            return p                                  # senza normalizzazione

        return F.normalize(p, dim=1)                            # post_proj

    def fill_cache(src_path, ids):
        for d in ("/data/cache_SAR", "/data/cache_OPT"):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        wanted = {f"{ID}.npy" for ID in ids}
        for modality, cache_dir in (("SAR", "/data/cache_SAR"), ("OPT", "/data/cache_OPT")):
            for part_path in sorted(glob(os.path.join(src_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, "r") as z:
                    for name in z.namelist():
                        if name in wanted:
                            z.extract(name, cache_dir)

    def embed(ids, nome):
        dataset = MoCo2encodersLoader(
            listIDs=ids, sar_map={}, opt_map={}, transform=None, patch_size=patch_size,
            n_images1=n_images1, n_channels1=n_channels1, n_images2=n_images2, n_channels2=n_channels2,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        out_q, out_k = [], []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for im_q, im_k in tqdm(loader, mininterval=30.0):       # im_q = ottico, im_k = sar
                out_q.append(branch_features(im_q.to(device), "q").float().flatten(1).cpu())
                out_k.append(branch_features(im_k.to(device), "k").float().flatten(1).cpu())
        q, k = torch.cat(out_q).numpy(), torch.cat(out_k).numpy()
        print(f"OK -> {nome}: {len(q)} serie, q {q.shape[1]} + k {k.shape[1]} = {q.shape[1] + k.shape[1]} dim", flush=True)
        return q, k

    print(f"Estrazione feature: {feature_point}", flush=True)
    fill_cache(normal_path, set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids))
    q_fit, k_fit = embed(normal_fit_ids, "normali fit")
    q_thr, k_thr = embed(normal_thr_ids, "normali soglia")
    q_normal, k_normal = embed(normal_test_ids, "normali test")

    fill_cache(anom_path, anom_ids)
    q_anom, k_anom = embed(anom_ids, "anomale")
    print(flush=True)


    z_fit    = np.concatenate([q_fit, k_fit], axis=1)
    z_thr    = np.concatenate([q_thr, k_thr], axis=1)
    z_normal = np.concatenate([q_normal, k_normal], axis=1)
    z_anom   = np.concatenate([q_anom, k_anom], axis=1)

    if standardize:
        scaler = StandardScaler().fit(z_fit)
        z_fit, z_thr, z_normal, z_anom = (scaler.transform(z) for z in (z_fit, z_thr, z_normal, z_anom))

    pca = PCA(n_components=pca_components, whiten=True).fit(z_fit.astype(np.float64))
    print(f"OK -> PCA: {pca.n_components_} componenti su {z_fit.shape[1]} dimensioni", flush=True)

    def md(z):
        return (pca.transform(z.astype(np.float64)) ** 2).sum(axis=1)

    res_thr, res_normal, res_anom = md(z_thr), md(z_normal), md(z_anom)


    auroc = roc_auc_score(np.r_[np.zeros(len(res_normal)), np.ones(len(res_anom))], np.r_[res_normal, res_anom])
    print(f"AUROC ({feature_point}): {auroc:.4f}", flush=True)

    area_anom = np.array([flood_area[ID] for ID in anom_ids])
    auroc_fasce = {}
    for lo, hi in ((0, 5), (5, 20), (20, 100)):
        sel = (area_anom > lo) & (area_anom <= hi)
        if sel.sum() > 0:
            a = roc_auc_score(np.r_[np.zeros(len(res_normal)), np.ones(sel.sum())], np.r_[res_normal, res_anom[sel]])
            auroc_fasce[f"{lo}-{hi}%"] = {"auroc": float(a), "n_anomale": int(sel.sum())}
            print(f"  AUROC fascia {lo}-{hi}%: {a:.4f} ({sel.sum()} anomale)", flush=True)
    print(flush=True)


    score_thr, score_normal, score_anom = -res_thr, -res_normal, -res_anom
    rules = {}
    for fa in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        thr = float(np.quantile(score_thr, fa))
        tp = int((score_anom < thr).sum());   fn = len(score_anom) - tp
        fp = int((score_normal < thr).sum()); tn = len(score_normal) - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        rules[f"score < FA {fa:.0%})"] = {
            "soglia": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp), "REC": rec, "PREC": prec,
            "F1": 2 * prec * rec / (prec + rec) if tp > 0 else 0.0,
        }
        print(f"FA {fa:.0%}: soglia {thr:.4f} | TP {tp} FP {fp} FN {fn} TN {tn} | "
              f"FA {fp/(fp+tn):.3f} REC {rec:.3f} PREC {prec:.3f}", flush=True)


    out_dir = f"/data/anomaly_results_mdf_{job_name}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    pd.DataFrame({
        "ID": normal_test_ids + anom_ids,
        "label": [0] * len(score_normal) + [1] * len(score_anom),
        "score": np.concatenate([score_normal, score_anom]),
        "flood_area": [0.0] * len(score_normal) + list(area_anom),
    }).to_csv(os.path.join(out_dir, "scores.csv"), index=False)

    if save_embeddings:
        np.savez_compressed(
            os.path.join(out_dir, "embeddings.npz"),
            fit_ids=np.array(normal_fit_ids), fit_q=q_fit, fit_k=k_fit,
            thr_ids=np.array(normal_thr_ids), thr_q=q_thr, thr_k=k_thr,
            normal_ids=np.array(normal_test_ids), normal_q=q_normal, normal_k=k_normal,
            anom_ids=np.array(anom_ids), anom_q=q_anom, anom_k=k_anom,
        )

    results = {
        "job_name": job_name, "metodo": "md", "feature_point": feature_point,
        "pca_components": pca_components, "n_componenti": int(pca.n_components_),
        "dataset_anomalies": dataset_anomalies, "moco_version": moco_version,
        "min_flood_area": min_flood_area, "standardize": standardize, "whitening": True,
        "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids),
        "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
        "auroc": float(auroc), "auroc_fasce": auroc_fasce, "rules": rules,
    }
    with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    shutil.make_archive(out_dir, "zip", out_dir)
    try:
        project_work.log_artifact(name=f"anomaly_results_mdf_{job_name}", source=f"{out_dir}.zip", kind="artifact")
        print(f"OK -> risultati salvati: anomaly_results_mdf_{job_name}", flush=True)
    except Exception as e:
        print(f"EXC -> upload risultati: {e}", flush=True)

    return "TERMINATO -> anomaly detection MD su feature"






@handler()
def ad_ocsvm_encoders(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",                 
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    ocsvm_nu: float = 0.05,
    ocsvm_gamma: str = "scale",
    standardize: bool = True,
    batch_size: int = 16,
    workers: int = 0,
    min_flood_area: float = 0.0,
    old_vectors: str = "",
    n_series_thr: int = 500,
    moco_attn: bool = False,                        
    weights_sar: str = "autoencoder_2Dattn_SAR_weights_v1",
    weights_opt: str = "autoencoder_2Dattn_OPT_weights_v1",
    sar_attn: bool = False,                         
    opt_attn: bool = False,
    sar_ltae: bool = False,
    opt_ltae: bool = False,
    sar_spac_net: bool = False,                     
    opt_spac_net: bool = False,
    latent_dim: int = 16,                           
    pool_grid: int = 4,                                                 
    pca_components: float | int | None = None,      
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    # lettura ids serie anomale
    print(f"Lettura serie anomale: {dataset_anomalies}", flush=True)
    anom_path = project_data.get_artifact(f"{dataset_anomalies}").download(f"/data/{dataset_anomalies}")
    ids_per_sensor_anom = {}

    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(anom_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_anom[sensor] = ids

    anom_ids = ids_per_sensor_anom["SAR"] & ids_per_sensor_anom["OPT"]
    num_fit_series = len(anom_ids)
    n_anom_tot = len(anom_ids)

    # filtro superficie alluvione
    if min_flood_area > 0:
        try:
            mask_parts = sorted(glob(os.path.join(anom_path, "MASK", "part*.zip")))

            flood_area = {}
            for part_path in mask_parts:
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('.npy') and name[:-4] in anom_ids:
                            mask = np.load(io.BytesIO(z.read(name)))
                            flood_area[name[:-4]] = 100.0 * float((mask == 2).sum()) / mask.size

            anom_ids = {ID for ID in anom_ids if flood_area.get(ID, 0.0) >= min_flood_area}
            print(f"OK -> flood >= {min_flood_area:g}%: {len(anom_ids)} serie su {n_anom_tot}", flush=True)

        except Exception as e:
            print(f"EXC -> flood: {e}", flush=True)

    anom_ids = sorted(anom_ids)
    print(f"OK -> {len(anom_ids)} serie anomale", "\n", flush=True)


    # lettura serie normali
    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(f"/data/Floods_{dataset_standard}_crop_norm")

    # import ids lits from moco training
    if moco_attn:
        p = project_work.get_artifact(f"moco_attn_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
    else:
        p = project_work.get_artifact(f"moco_2D_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)

    with open(p, 'r') as f:
        test_ids = json.load(f)
    print(f"OK -> liste normali caricate: {len(test_ids)} test", flush=True)
    num_fit_series = len(test_ids) - num_fit_series

    random.seed(0)
    shuffled_test_ids = random.sample(test_ids, len(test_ids))

    ids_per_sensor_normal = {}

    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(normal_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_normal[sensor] = ids

    normal_ids = ids_per_sensor_normal["SAR"] & ids_per_sensor_normal["OPT"]

    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID in normal_ids]

    normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]                            # fit -> iperpiano
    normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]              # calcolo soglie
    normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]             # anomaly detection


    print(f"OK -> {len(test_ids)} serie normali tot, {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali")

    if old_vectors:
        print(f"Upload embedding: anomaly_results_ocsvm_enc_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_ocsvm_enc_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            z.extract("embeddings.npz", "/data/old_vectors")
            old_results = json.loads(z.read("anomaly_results.json"))

        old = np.load("/data/old_vectors/embeddings.npz")
        lookup = {}
        for g in ("fit", "thr", "normal", "anom"):
            for i, ID in enumerate(old[f"{g}_ids"]):
                lookup[str(ID)] = (old[f"{g}_q"][i], old[f"{g}_k"][i])

        def take(ids, nome):
            missing = [ID for ID in ids if ID not in lookup]
            if missing:
                raise ValueError(f"{len(missing)} serie {nome} non presenti in {old_vectors} (es. {missing[:3]})")
            return np.stack([lookup[ID][0] for ID in ids]), np.stack([lookup[ID][1] for ID in ids])

        q_fit, k_fit = take(normal_fit_ids, "fit")
        q_thr, k_thr = take(normal_thr_ids, "thr")
        q_normal, k_normal = take(normal_test_ids, "normali test")
        q_anom, k_anom = take(anom_ids, "anomale")
        print(f"OK -> embedding caricatu: {len(q_fit)} fit,  {len(q_thr)} thr, {len(q_normal)} normali, {len(q_anom)} anomale", "\n", flush=True)

    else:

        print("calcolo vettori encoder", flush = True)

        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"

        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids)}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)

        # sar
        if sar_attn:
            caeSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=latent_dim, n_images=n_images1,
                                          n_head=8, d_k=8, ltae=sar_ltae, spac_net=sar_spac_net).to(device)
        else:
            caeSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=latent_dim, n_images=n_images1,
                                        n_head=8, d_k=8, ltae=sar_ltae).to(device)

        w_path = project_work.get_artifact(weights_sar).download(f"/data/{weights_sar}.pth", overwrite=True)
        state_dict = torch.load(w_path, map_location=device)
        if all(k.startswith('module.') for k in state_dict.keys()):              
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        caeSAR.load_state_dict(state_dict)
        caeSAR.eval()
        encoderSAR = caeSAR.encoder
        print(f"OK -> autoencoder SAR caricato: {weights_sar}", flush=True)

        # opt
        if opt_attn:
            caeOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=latent_dim, n_images=n_images2,
                                          n_head=8, d_k=8, ltae=opt_ltae, spac_net=opt_spac_net).to(device)
        else:
            caeOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=latent_dim, n_images=n_images2,
                                        n_head=8, d_k=8, ltae=opt_ltae).to(device)

        w_path = project_work.get_artifact(weights_opt).download(f"/data/{weights_opt}.pth", overwrite=True)
        state_dict = torch.load(w_path, map_location=device)
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        caeOPT.load_state_dict(state_dict)
        caeOPT.eval()
        encoderOPT = caeOPT.encoder
        print(f"OK -> autoencoder ottico caricato: {weights_opt}", "\n", flush=True)

        pool_layer = nn.AdaptiveAvgPool2d(pool_grid)
        
        # embedding encoder: pesi congelati
        # 2 matrici: rows = serie, cols = dim
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            out_q, out_k = [], []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)
                    q = pool_layer(encoderOPT(im_q)).flatten(1)
                    k = pool_layer(encoderSAR(im_k)).flatten(1)
                    out_q.append(q.float().flatten(1).cpu())
                    out_k.append(k.float().flatten(1).cpu())
            return torch.cat(out_q).numpy(), torch.cat(out_k).numpy()

        # embedding serie di fit
        print("Embedding encoder: serie di fit ocsvm", flush=True)
        q_fit, k_fit = embed(normal_fit_ids)
        print(f"OK -> normali fit: {len(q_fit)} serie, q {q_fit.shape[1]} + k {k_fit.shape[1]} = {q_fit.shape[1] + k_fit.shape[1]} dim", "\n", flush=True)

        # embedding serie thr
        print("Embedding encoder: serie thr ocsvm", flush=True)
        q_thr, k_thr = embed(normal_thr_ids)
        print(f"OK -> normali thr: {len(q_thr)} serie, q {q_thr.shape[1]} + k {k_thr.shape[1]} = {q_thr.shape[1] + k_thr.shape[1]} dim", "\n", flush=True)

        # embedding serie normali
        print("Embedding encoder: serie normali", flush=True)
        q_normal, k_normal = embed(normal_test_ids)
        print(f"OK -> normali test: {len(q_normal)} serie, q {q_normal.shape[1]} + k {k_normal.shape[1]} = {q_normal.shape[1] + k_normal.shape[1]} dim", "\n", flush=True)

        # pulizia cartelle
        for d in ("/data/cache_SAR", "/data/cache_OPT"):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        # estrazione serie anomale
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_anom_series = {f"{ID}.npy" for ID in anom_ids}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(anom_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_anom_series:
                            z.extract(name, cache_dir)

        # embedding serie anomale
        print("Embedding encoder: serie anomale", flush=True)
        q_anom, k_anom = embed(anom_ids)
        print(f"OK -> anomale: {len(q_anom)} serie, q {q_anom.shape[1]} + k {k_anom.shape[1]} = {q_anom.shape[1] + k_anom.shape[1]} dim", "\n", flush=True)

    # concatenazione
    w_fit = np.concatenate([q_fit, k_fit], axis=1).astype(np.float64)
    w_thr = np.concatenate([q_thr, k_thr], axis=1).astype(np.float64)
    w_normal = np.concatenate([q_normal, k_normal], axis=1).astype(np.float64)
    w_anom = np.concatenate([q_anom, k_anom],axis=1).astype(np.float64)

    if standardize:
        scaler = StandardScaler().fit(w_fit)
        w_fit, w_thr, w_normal, w_anom = scaler.transform(w_fit), scaler.transform(w_thr), scaler.transform(w_normal), scaler.transform(w_anom)

    pca = PCA(whiten=True).fit(w_fit)       
    z_fit = pca.transform(w_fit)
    z_thr = pca.transform(w_thr)
    z_normal = pca.transform(w_normal)
    z_anom = pca.transform(w_anom)

    gamma = ocsvm_gamma if ocsvm_gamma in ("scale", "auto") else float(ocsvm_gamma)
    print(f"nu={ocsvm_nu}, gamma={gamma}, standardize={standardize})", flush=True)

    ocsvm = OneClassSVM(kernel="rbf", nu=ocsvm_nu, gamma=gamma).fit(z_fit)

    # punteggi: decision_function > 0 dentro il confine , < 0 fuori
    score_fit = ocsvm.decision_function(z_fit).astype(np.float64)
    score_thr = ocsvm.decision_function(z_thr).astype(np.float64)
    score_normal = ocsvm.decision_function(z_normal).astype(np.float64)
    score_anom = ocsvm.decision_function(z_anom).astype(np.float64)

    y = np.r_[np.zeros(len(score_normal)), np.ones(len(score_anom))]
    auroc = roc_auc_score(y, -np.r_[score_normal, score_anom])

    frac_outside_fit = float((score_fit < 0).mean())
    print(f"auroc: {auroc:.4f} | fuori dal confine (fit): {frac_outside_fit:.3f}", flush=True)

    # salvataggio e risultati
    out_dir = f"/data/anomaly_results_ocsvm_enc_{job_name}"           
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    # punteggi
    df = pd.DataFrame({
        "ID": normal_test_ids + anom_ids,
        "label": [0] * len(score_normal) + [1] * len(score_anom),
        "score": np.concatenate([score_normal, score_anom]),
    })
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    pd.DataFrame({"ID": normal_thr_ids, "score": score_thr}).to_csv(os.path.join(out_dir, "scores_thr.csv"), index=False)   # nuovo


    # soglia
    false_allarms = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    thr_scores = score_thr
    eval_norm = score_normal
    anom = df[df.label == 1].score.values

    rules = {}
    for fa in false_allarms:
        thr = float(np.quantile(thr_scores, fa))
        tp = int((anom < thr).sum()); fn = len(anom) - tp
        fp = int((eval_norm < thr).sum()); tn = len(eval_norm) - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)

        rules[f"FA {fa:.0%}"] = {
            "soglia": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp), "REC": rec, "PREC": prec,
            "F1": 2 * prec * rec / (prec + rec) if tp > 0 else 0.0,
        }
        print(f"FA {fa:.0%}: soglia {thr:.4f} | TP {tp} FP {fp} FN {fn} TN {tn} | FA {fp/(fp+tn):.3f} REC {rec:.3f} PREC {prec:.3f}", flush=True)

    # vettori
    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        fit_ids=np.array(normal_fit_ids), fit_q=q_fit, fit_k=k_fit,
        thr_ids=np.array(normal_thr_ids), thr_q=q_thr, thr_k=k_thr,
        normal_ids=np.array(normal_test_ids), normal_q=q_normal, normal_k=k_normal,
        anom_ids=np.array(anom_ids), anom_q=q_anom, anom_k=k_anom,
    )

    # riassunto
    results = {
        "job_name": job_name, "metodo": "ocsvm", "dataset_anomalies": dataset_anomalies,
        "moco_version": moco_version, "min_flood_area": min_flood_area, "standardize": standardize,
        "weights_sar": weights_sar, "weights_opt": weights_opt,
        "pool_size": pool_grid, "pca_components": pca_components, "n_componenti": int(pca.n_components_),
        "ocsvm": {"nu": ocsvm_nu, "gamma": gamma, "dim": int(z_fit.shape[1]), "frac_outside_fit": frac_outside_fit},
        "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
        "auroc": float(auroc),
        "rules": rules,
    }
    with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    shutil.make_archive(out_dir, "zip", out_dir)
    project_work.log_artifact(name=f"anomaly_results_ocsvmenc_{job_name}", source=f"{out_dir}.zip", kind="artifact")    # MODIFICATO

    return "TERMINATO -> anomaly detection ocsvm encoder"



@handler()
def ad_md_encoders(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",                 
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    standardize: bool = True,
    batch_size: int = 16,
    workers: int = 0,
    min_flood_area: float = 0.0,
    old_vectors: str = "",
    n_series_thr: int = 500,
    moco_attn: bool = False,                        
    weights_sar: str = "autoencoder_2Dattn_SAR_weights_v1",
    weights_opt: str = "autoencoder_2Dattn_OPT_weights_v1",
    sar_attn: bool = False,                         
    opt_attn: bool = False,
    sar_ltae: bool = False,
    opt_ltae: bool = False,
    sar_spac_net: bool = False,                     
    opt_spac_net: bool = False,
    latent_dim: int = 16,                           
    pool_grid: int = 4,                                                 
    pca_components: float | int | None = None,               
):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
  
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    # lettura ids serie anomale
    print(f"Lettura serie anomale: {dataset_anomalies}", flush=True)
    anom_path = project_data.get_artifact(f"{dataset_anomalies}").download(f"/data/{dataset_anomalies}")
    ids_per_sensor_anom = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(anom_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_anom[sensor] = ids
 
    anom_ids = ids_per_sensor_anom["SAR"] & ids_per_sensor_anom["OPT"]
    num_fit_series = len(anom_ids)
 
    # filtro superficie alluvione
    if min_flood_area > 0:
        try:
            mask_parts = sorted(glob(os.path.join(anom_path, "MASK", "part*.zip")))

            flood_area = {}
            for part_path in mask_parts:
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('.npy') and name[:-4] in anom_ids:
                            mask = np.load(io.BytesIO(z.read(name)))
                            flood_area[name[:-4]] = 100.0 * float((mask == 2).sum()) / mask.size

            anom_ids = {ID for ID in anom_ids if flood_area.get(ID, 0.0) >= min_flood_area}
            print(f"OK -> flood >= {min_flood_area:g}%: {len(anom_ids)} serie su {len(anom_ids)}", flush=True)

        except Exception as e:
            print(f"EXC -> flood: {e}", flush=True)
 
    anom_ids = sorted(anom_ids)
    print(f"OK -> {len(anom_ids)} serie anomale", "\n", flush=True)


    # lettura serie normali
    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(f"/data/Floods_{dataset_standard}_crop_norm")

    # import ids lits from moco training
    if moco_attn:
        p = project_work.get_artifact(f"moco_attn_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
    else:
        p = project_work.get_artifact(f"moco_2D_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)

    with open(p, 'r') as f:
        test_ids = json.load(f)
    print(f"OK -> liste normali caricate: {len(test_ids)} test", flush=True)
    num_fit_series = len(test_ids) - num_fit_series
 
    random.seed(0)
    shuffled_test_ids = random.sample(test_ids, len(test_ids)) 

    ids_per_sensor_normal = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(normal_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_normal[sensor] = ids
 
    normal_ids = ids_per_sensor_normal["SAR"] & ids_per_sensor_normal["OPT"]
 
    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID in normal_ids]

    normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]                            # fit -> var
    normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]              # calcolo soglie
    normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]             # anomaly detection 


    print(f"OK -> {len(test_ids)} serie normali tot, {len(normal_fit_ids)} serie fit, {len(normal_test_ids)} serie test normali")

    if old_vectors:
        print(f"Upload embedding: anomaly_results_md_enc_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_md_enc_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            z.extract("embeddings.npz", "/data/old_vectors")
            old_results = json.loads(z.read("anomaly_results.json"))

        old = np.load("/data/old_vectors/embeddings.npz")
        lookup = {}
        for g in ("fit", "thr", "normal", "anom"):
            for i, ID in enumerate(old[f"{g}_ids"]):
                lookup[str(ID)] = (old[f"{g}_q"][i], old[f"{g}_k"][i])

        def take(ids, nome):
            missing = [ID for ID in ids if ID not in lookup]
            if missing:
                raise ValueError(f"{len(missing)} serie {nome} non presenti in {old_vectors} (es. {missing[:3]})")
            return np.stack([lookup[ID][0] for ID in ids]), np.stack([lookup[ID][1] for ID in ids])

        q_fit, k_fit = take(normal_fit_ids, "fit")
        q_thr, k_thr = take(normal_thr_ids, "thr")
        q_normal, k_normal = take(normal_test_ids, "normali test")
        q_anom, k_anom = take(anom_ids, "anomale")
        print(f"OK -> embedding caricatu: {len(q_fit)} fit,  {len(q_thr)} thr, {len(q_normal)} normali, {len(q_anom)} anomale", "\n", flush=True)

    else:

        print("calcolo vettori MoCo", flush = True)

        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"

        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids)}       
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)

        # sar
        if sar_attn:
            caeSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=latent_dim, n_images=n_images1,
                                          n_head=8, d_k=8, ltae=sar_ltae, spac_net=sar_spac_net).to(device)
        else:
            caeSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=latent_dim, n_images=n_images1,
                                        n_head=8, d_k=8, ltae=sar_ltae).to(device)

        w_path = project_work.get_artifact(weights_sar).download(f"/data/{weights_sar}.pth", overwrite=True)
        state_dict = torch.load(w_path, map_location=device)
        if all(k.startswith('module.') for k in state_dict.keys()):              
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        caeSAR.load_state_dict(state_dict)
        caeSAR.eval()
        encoderSAR = caeSAR.encoder
        print(f"OK -> autoencoder SAR caricato: {weights_sar}", flush=True)

        # opt
        if opt_attn:
            caeOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=latent_dim, n_images=n_images2,
                                          n_head=8, d_k=8, ltae=opt_ltae, spac_net=opt_spac_net).to(device)
        else:
            caeOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=latent_dim, n_images=n_images2,
                                        n_head=8, d_k=8, ltae=opt_ltae).to(device)

        w_path = project_work.get_artifact(weights_opt).download(f"/data/{weights_opt}.pth", overwrite=True)
        state_dict = torch.load(w_path, map_location=device)
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        caeOPT.load_state_dict(state_dict)
        caeOPT.eval()
        encoderOPT = caeOPT.encoder
        print(f"OK -> autoencoder ottico caricato: {weights_opt}", "\n", flush=True)

        pool_layer = nn.AdaptiveAvgPool2d(pool_grid)

        # embedding encoder: pesi congelati
        # 2 matrici: rows = serie, cols = dim
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            out_q, out_k = [], []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)
                    q = pool_layer(encoderOPT(im_q)).flatten(1)
                    k = pool_layer(encoderSAR(im_k)).flatten(1)
                    out_q.append(q.float().flatten(1).cpu())
                    out_k.append(k.float().flatten(1).cpu())
            return torch.cat(out_q).numpy(), torch.cat(out_k).numpy()

        # embedding serie di fit 
        print("Embedding MoCo: serie di fit pac", flush=True)
        q_fit, k_fit = embed(normal_fit_ids)
        print(f"OK -> normali fit: {len(q_fit)} serie, q {q_fit.shape[1]} + k {k_fit.shape[1]} = {q_fit.shape[1] + k_fit.shape[1]} dim", "\n", flush=True)

        # embedding serie thr 
        print("Embedding MoCo: serie thr pac", flush=True)
        q_thr, k_thr = embed(normal_thr_ids)
        print(f"OK -> normali thr: {len(q_thr)} serie, q {q_thr.shape[1]} + k {k_thr.shape[1]} = {q_thr.shape[1] + k_thr.shape[1]} dim", "\n", flush=True)
 
        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        q_normal, k_normal = embed(normal_test_ids)
        print(f"OK -> normali test: {len(q_normal)} serie, q {q_normal.shape[1]} + k {k_normal.shape[1]} = {q_normal.shape[1] + k_normal.shape[1]} dim", "\n", flush=True)

        # pulizia cartelle
        for d in ("/data/cache_SAR", "/data/cache_OPT"):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        # estrazione serie anomale
        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_anom_series = {f"{ID}.npy" for ID in anom_ids}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(anom_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_anom_series:
                            z.extract(name, cache_dir)

        # embedding serie anomale
        print("Embedding MoCo: serie anomale", flush=True)
        q_anom, k_anom = embed(anom_ids)
        print(f"OK -> anomale: {len(q_anom)} serie, q {q_anom.shape[1]} + k {k_anom.shape[1]} = {q_anom.shape[1] + k_anom.shape[1]} dim", "\n", flush=True)
 
    # concatenazione
    z_fit = np.concatenate([q_fit, k_fit], axis=1)
    z_thr = np.concatenate([q_thr, k_thr], axis=1)
    z_normal = np.concatenate([q_normal, k_normal], axis=1)
    z_anom = np.concatenate([q_anom, k_anom],axis=1)

    if standardize:
        scaler = StandardScaler().fit(z_fit)
        z_fit, z_thr, z_normal, z_anom = scaler.transform(z_fit), scaler.transform(z_thr), scaler.transform(z_normal), scaler.transform(z_anom)

    # distanza mahalanobis
    pca = PCA(whiten=True).fit(z_fit.astype(np.float64))
    res_thr = (pca.transform(z_thr.astype(np.float64)) ** 2).sum(axis=1)
    res_normal = (pca.transform(z_normal.astype(np.float64)) ** 2).sum(axis=1)
    res_anom = (pca.transform(z_anom.astype(np.float64)) ** 2).sum(axis=1)

    y = np.r_[np.zeros(len(res_normal)), np.ones(len(res_anom))]
    print(roc_auc_score(y, np.r_[res_normal, res_anom]))

    # salvataggio e risultati
    out_dir = f"/data/anomaly_results_md_enc_{job_name}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    
    # punteggi
    score_normal = -res_normal.astype(np.float64)
    score_anom = -res_anom.astype(np.float64)
    score_thr = -res_thr.astype(np.float64)
    auroc = roc_auc_score(np.r_[np.zeros(len(res_normal)), np.ones(len(res_anom))], np.r_[res_normal, res_anom])
    
    df = pd.DataFrame({
        "ID": normal_test_ids + anom_ids,
        "label": [0] * len(score_normal) + [1] * len(score_anom),
        "score": np.concatenate([score_normal, score_anom]),
    })
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    
    # soglia
    false_allarms = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)                             
    thr_scores = score_thr        
    eval_norm = score_normal                          
    anom = df[df.label == 1].score.values
    
    rules = {}
    for fa in false_allarms:
        thr = float(np.quantile(thr_scores, fa))
        tp = int((anom < thr).sum()); fn = len(anom) - tp
        fp = int((eval_norm < thr).sum()); tn = len(eval_norm) - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)

        rules[f"FA {fa:.0%}"] = {
            "soglia": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp), "REC": rec, "PREC": prec,
            "F1": 2 * prec * rec / (prec + rec) if tp > 0 else 0.0,
        }
        print(f"FA {fa:.0%}: soglia {thr:.4f} | TP {tp} FP {fp} FN {fn} TN {tn} | FA {fp/(fp+tn):.3f} REC {rec:.3f} PREC {prec:.3f}", flush=True)
    
    # vettori 
    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        fit_ids=np.array(normal_fit_ids), fit_q=q_fit, fit_k=k_fit,
        thr_ids=np.array(normal_thr_ids), thr_q=q_thr, thr_k=k_thr,                                 
        normal_ids=np.array(normal_test_ids), normal_q=q_normal, normal_k=k_normal,
        anom_ids=np.array(anom_ids), anom_q=q_anom, anom_k=k_anom,
    )
    
    # riassunto
    results = {
        "job_name": job_name, "metodo": "md", "dataset_anomalies": dataset_anomalies,
        "moco_version": moco_version, "min_flood_area": min_flood_area, "standardize": standardize,
        "weights_sar": weights_sar, "weights_opt": weights_opt,
        "whitening": True, "n_componenti": int(pca.n_components_),
        "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
        "auroc": float(auroc),
        "rules": rules,
    }
    with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    shutil.make_archive(out_dir, "zip", out_dir)
    project_work.log_artifact(name=f"anomaly_results_mdenc_{job_name}", source=f"{out_dir}.zip", kind="artifact")














