import os
import zipfile
import shutil
import torch
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

from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_2d
from floodlib.loader import MoCo2encodersLoader
from floodlib.builder import MoCo2encoders_2d


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
    old_vectors: str = ""                   
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
    normal_fit_ids = shuffled_test_ids[:num_fit_series]                                             # fit -> var
    normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]              # anomaly detection 


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
        print(f"OK -> embedding caricati: {len(q_fit)} fit, {len(q_normal)} normali, {len(q_anom)} anomale", "\n", flush=True)

    else:

        print("calcolo vettori MoCo", flush = True)

        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"

        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_test_ids)}
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)

        # caricamento pesi moco
        try:
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

    if standardize:
        scaler = StandardScaler().fit(z_fit)
        z_fit, z_normal, z_anom = scaler.transform(z_fit), scaler.transform(z_normal), scaler.transform(z_anom)

    # distanza mahalanobis
    pca = PCA(whiten=True).fit(z_fit.astype(np.float64))
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
    auroc = roc_auc_score(np.r_[np.zeros(len(res_normal)), np.ones(len(res_anom))], np.r_[res_normal, res_anom])
    
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
        rules[f"md score < soglia (FA {fa:.0%})"] = {
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
    )
    
    # riassunto
    results = {
        "job_name": job_name, "metodo": "md", "dataset_anomalies": dataset_anomalies,
        "moco_version": moco_version, "min_flood_area": min_flood_area, "standardize": standardize,
        "whitening": True, "n_componenti": int(pca.n_components_),
        "n_fit": len(normal_fit_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
        "auroc": float(auroc),
        "rules": rules,
    }
    with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    shutil.make_archive(out_dir, "zip", out_dir)
    project_work.log_artifact(name=f"anomaly_results_md_{job_name}", source=f"{out_dir}.zip", kind="artifact")
















