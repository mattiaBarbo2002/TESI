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
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from digitalhub_runtime_python import handler
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata, spearmanr

from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_2d
from floodlib.loader import MoCo2encodersLoader
from floodlib.builder import MoCo2encoders_2d

matplotlib.use("Agg") 


# notebook -> anomaly_detection

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
    n_normal_eval: int = 2402,                     
    n_threshold_series: int = 0,                    
    num_anomalies: int = 100,
    n_std: float = 1.0,                             
    batch_size: int = 32,
    workers: int = 0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
 
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    # import ids lits from moco training
    try:
        p = project_work.get_artifact(f"moco_2D_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
        with open(p, 'r') as f:
            train_ids = json.load(f)
        p = project_work.get_artifact(f"moco_2D_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
        with open(p, 'r') as f:
            test_ids = json.load(f)
        print(f"OK -> liste normali caricate: {len(train_ids)} train, {len(test_ids)} test", flush=True)
 
    except Exception as e:
        print(f"EXC -> caricamento liste ID normali: {e}", flush=True)
 
    random.seed(0)
    shuffled_test_ids = random.sample(test_ids, len(test_ids))     # tutte le serie normali vengono da testIdsList (mai viste da MoCo in training)
 
    # lettura serie normali
    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(f"/data/Floods_{dataset_standard}_crop_norm")
    ids_per_sensor_normal = {}
    
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(normal_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_normal[sensor] = ids
 
    normal_ids = ids_per_sensor_normal["SAR"] & ids_per_sensor_normal["OPT"]
 
    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID in normal_ids]
    if len(shuffled_test_ids) <= n_normal_eval:
        raise ValueError(f"serie normali di testIdsList disponibili: {len(shuffled_test_ids)}, ne servono piu' di "
                         f"n_normal_eval={n_normal_eval} per avere anche serie per la soglia")
    normal_test_ids = shuffled_test_ids[:n_normal_eval]            # anomaly detection (normali vs anomale)
    normal_thr_ids = shuffled_test_ids[n_normal_eval:]             # soglia: le restanti, mai usate nella valutazione
    if n_threshold_series > 0:
        normal_thr_ids = normal_thr_ids[:n_threshold_series]
    print(f"OK -> {len(normal_thr_ids)} serie soglia, {len(normal_test_ids)} serie test normali "
          f"(tutte da testIdsList, nessuna in comune)", flush=True)
 
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
 
    anom_ids = sorted(anom_ids)
    random.seed(1)
    anom_ids = random.sample(anom_ids, num_anomalies)
    print(f"OK -> {len(anom_ids)} serie anomale", "\n", flush=True)
 
    # estrazione serie normal_ids, threshold_ids + test_ad_ids
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
      
 
    # esecuzione moco pesi congelati -> calcolo soglia 
    print("Cosine similarity: soglia", flush=True)
    dataset_thr = MoCo2encodersLoader(
            listIDs=normal_thr_ids, sar_map={}, opt_map={}, transform=None,
            patch_size=patch_size,
            n_images1=n_images1, n_channels1=n_channels1,
            n_images2=n_images2, n_channels2=n_channels2,
        )
    loader_thr = DataLoader(dataset_thr, batch_size=batch_size, shuffle=False, num_workers=workers)
 
    out = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for im_q, im_k in tqdm(loader_thr, mininterval=30.0):   # im_q = OPT, im_k = SAR
            im_q = im_q.to(device)
            im_k = im_k.to(device)
            _, q, k = model.contrastive_loss(im_q, im_k)
            out.append((q.float() * k.float()).sum(dim=1).cpu())
    cos_sim_thr = torch.cat(out).numpy()
    print(f"OK -> soglia: {len(cos_sim_thr)} similarita' calcolate", "\n", flush=True)
 
    # esecuzione moco pesi congelati -> calcolo cos sim serie normali
    print("Cosine similarity: serie normali", flush=True)
    dataset_normal = MoCo2encodersLoader(
            listIDs=normal_test_ids, sar_map={}, opt_map={}, transform=None,
            patch_size=patch_size,
            n_images1=n_images1, n_channels1=n_channels1,
            n_images2=n_images2, n_channels2=n_channels2,
        )
    loader_normal = DataLoader(dataset_normal, batch_size=batch_size, shuffle=False, num_workers=workers)
 
    out = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for im_q, im_k in tqdm(loader_normal, mininterval=30.0):   # im_q = OPT, im_k = SAR
            im_q = im_q.to(device)
            im_k = im_k.to(device)
            _, q, k = model.contrastive_loss(im_q, im_k)
            out.append((q.float() * k.float()).sum(dim=1).cpu())
    cos_sim_normal = torch.cat(out).numpy()
    print(f"OK -> serie normali: {len(cos_sim_normal)} similarita' calcolate", "\n", flush=True)
 
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
 
    # esecuzione moco pesi congelati -> calcolo cos sim serie anomale
    print("Cosine similarity: serie anomale", flush=True)
    dataset_anom = MoCo2encodersLoader(
            listIDs=anom_ids, sar_map={}, opt_map={}, transform=None,
            patch_size=patch_size,
            n_images1=n_images1, n_channels1=n_channels1,
            n_images2=n_images2, n_channels2=n_channels2,
        )
    loader_anom = DataLoader(dataset_anom, batch_size=batch_size, shuffle=False, num_workers=workers)
 
    out = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for im_q, im_k in tqdm(loader_anom, mininterval=30.0):   # im_q = OPT, im_k = SAR
            im_q = im_q.to(device)
            im_k = im_k.to(device)
            _, q, k = model.contrastive_loss(im_q, im_k)
            out.append((q.float() * k.float()).sum(dim=1).cpu())
    cos_sim_anom = torch.cat(out).numpy()
    print(f"OK -> serie anomale: {len(cos_sim_anom)} similarita' calcolate", "\n", flush=True)
 
    # calcolo risultati, y=label -> 0=
    thr_mean = float(cos_sim_thr.mean())
    thr_std = float(cos_sim_thr.std())
    thr_high = thr_mean + (n_std * thr_std)
    thr_low = thr_mean - (n_std * thr_std)
 
    scores = np.concatenate([cos_sim_normal, cos_sim_anom])
    label = np.concatenate([np.zeros(len(cos_sim_normal)), np.ones(len(cos_sim_anom))])
 
    ranks = pd.Series(scores).rank().values
    auc_high = float((ranks[len(cos_sim_normal):].sum() - len(cos_sim_anom) * (len(cos_sim_anom) + 1) / 2) / (len(cos_sim_anom) * len(cos_sim_normal)))
    auc_low = 1.0 - auc_high
 
    tests = {
        f"cos_sim > m+{n_std:g}*std": scores > thr_high,
        f"cos_sim < m-{n_std:g}*std": scores < thr_low,
        f"m-{n_std:g}*std < cos_sim < m+{n_std:g}*std": (scores > thr_high) | (scores < thr_low),
    }
 
    tests_metrics = {}
    for key, value in tests.items():
        tp = int((value & (label == 1)).sum())
        fp = int((value & (label == 0)).sum())
        fn = int((~value & (label == 1)).sum())
        tn = int((~value & (label == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        rec = tp / (tp + fn)
        
        tests_metrics[key] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp),
            "REC": rec, "PREC": prec,
            "F1": (2 * prec * rec / (prec + rec)) if (tp > 0) else 0.0,
        }
 
    results = {
        "job_name": job_name,
        "dataset_anomalies": dataset_anomalies,
        "n_thr": len(cos_sim_thr), "n_normali": len(cos_sim_normal), "n_anomale": len(cos_sim_anom),
        "thr": {"mean": thr_mean, "std": thr_std, "n_std": n_std, "thr_high": thr_high, "thr_low": thr_low},
        "cos_sim_normal": {"mean": float(cos_sim_normal.mean()), "std": float(cos_sim_normal.std())},
        "cos_sim_anom": {"mean": float(cos_sim_anom.mean()), "std": float(cos_sim_anom.std())},
        "auroc_cos_sim_high_anom": auc_high,
        "auroc_cos_sim_low_anom": auc_low,
        "rules": tests_metrics,
    }
 
    # risultati
 
    out_dir = f"/data/anomaly_{job_name}"
    os.makedirs(out_dir, exist_ok=True)
 
    pd.DataFrame({
        "ID": normal_test_ids + anom_ids,
        "label": [0] * len(cos_sim_normal) + [1] * len(cos_sim_anom),
        "similarity": np.concatenate([cos_sim_normal, cos_sim_anom]),
    }).to_csv(os.path.join(out_dir, "similarities.csv"), index=False)
    with open(os.path.join(out_dir, "anomaly_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)   
 
    try:
        shutil.make_archive(out_dir, 'zip', out_dir)
        project_work.log_artifact(name=f"anomaly_results_{job_name}", source=f"{out_dir}.zip", kind='artifact')
        print("OK -> risultati salvati", flush=True)
    except Exception as e:
        print(f"EXC -> upload risultati: {e}", flush=True)
 
    return "TERMINATO -> anomaly detection"



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
    n_normal_eval: int = 2402,                      # serie normali (da testIdsList) usate nell'anomaly detection, di solito = num_anomalies
    n_fit_series: int = 0,                          # serie normali per addestrare l'OC-SVM: 0 = tutte le restanti di testIdsList; N > 0 = al massimo N
    num_anomalies: int = 100,
    ocsvm_nu: float = 0.05,                         # quota di serie di fit che l'OC-SVM puo' lasciare fuori dal confine
    ocsvm_gamma: str = "scale",                     # "scale" | "auto" | numero (es. "0.01"): ampiezza del kernel RBF
    standardize: bool = True,                       # StandardScaler (fit sulle serie di fit) sul vettore concatenato [q; k]
    batch_size: int = 32,
    workers: int = 0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
 
    # sklearn subito, cosi' se manca si ferma prima dei download e delle estrazioni

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    # import ids lits from moco training
    try:
        p = project_work.get_artifact(f"moco_2D_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
        with open(p, 'r') as f:
            train_ids = json.load(f)
        p = project_work.get_artifact(f"moco_2D_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
        with open(p, 'r') as f:
            test_ids = json.load(f)
        print(f"OK -> liste normali caricate: {len(train_ids)} train, {len(test_ids)} test", flush=True)
 
    except Exception as e:
        print(f"EXC -> caricamento liste ID normali: {e}", flush=True)
 
    random.seed(0)
    shuffled_test_ids = random.sample(test_ids, len(test_ids))     # tutte le serie normali vengono da testIdsList (mai viste da MoCo in training)
 
    # lettura serie normali
    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(f"/data/Floods_{dataset_standard}_crop_norm")
    ids_per_sensor_normal = {}
 
    for sensor in ("SAR", "OPT"):
        ids = set()
        for part_path in sorted(glob(os.path.join(normal_path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part_path, 'r') as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
        ids_per_sensor_normal[sensor] = ids
 
    normal_ids = ids_per_sensor_normal["SAR"] & ids_per_sensor_normal["OPT"]
 
    shuffled_test_ids = [ID for ID in shuffled_test_ids if ID in normal_ids]
    if len(shuffled_test_ids) <= n_normal_eval:
        raise ValueError(f"serie normali di testIdsList disponibili: {len(shuffled_test_ids)}, ne servono piu' di "
                         f"n_normal_eval={n_normal_eval} per avere anche serie per addestrare l'OC-SVM")
    normal_test_ids = shuffled_test_ids[:n_normal_eval]            # anomaly detection (normali vs anomale)
    normal_fit_ids = shuffled_test_ids[n_normal_eval:]             # fit OC-SVM: le restanti, mai usate nella valutazione
    if n_fit_series > 0:
        normal_fit_ids = normal_fit_ids[:n_fit_series]
    print(f"OK -> {len(normal_fit_ids)} serie fit OC-SVM, {len(normal_test_ids)} serie test normali "
          f"(tutte da testIdsList, nessuna in comune)", flush=True)
 
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
 
    anom_ids = sorted(anom_ids)
    random.seed(1)
    anom_ids = random.sample(anom_ids, num_anomalies)
    print(f"OK -> {len(anom_ids)} serie anomale", "\n", flush=True)
 
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
 
    # embedding MoCo a pesi congelati: q (ottico) e k (SAR), un vettore per serie nell'ordine di ids
    def _embed(ids):
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
 
    # embedding serie di fit (addestrano l'OC-SVM)
    print("Embedding MoCo: serie di fit OC-SVM", flush=True)
    q_fit, k_fit = _embed(normal_fit_ids)
    print(f"OK -> fit: {len(q_fit)} serie, q {q_fit.shape[1]} + k {k_fit.shape[1]} = {q_fit.shape[1] + k_fit.shape[1]} dimensioni", "\n", flush=True)
 
    # embedding serie normali
    print("Embedding MoCo: serie normali", flush=True)
    q_normal, k_normal = _embed(normal_test_ids)
    print(f"OK -> serie normali: {len(q_normal)} serie", "\n", flush=True)
 
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
    q_anom, k_anom = _embed(anom_ids)
    print(f"OK -> serie anomale: {len(q_anom)} serie", "\n", flush=True)
 
    # OC-SVM sul vettore concatenato [q (OPT); k (SAR)], addestrato solo su serie normali
    z_fit = np.concatenate([q_fit, k_fit], axis=1)
    scaler = StandardScaler().fit(z_fit) if standardize else None
 
    def _prep(q, k):
        z = np.concatenate([q, k], axis=1)
        return scaler.transform(z) if scaler is not None else z
 
    gamma = ocsvm_gamma if ocsvm_gamma in ("scale", "auto") else float(ocsvm_gamma)
    print(f"OC-SVM: fit su {len(z_fit)} serie (nu={ocsvm_nu}, gamma={gamma}, standardize={standardize})", flush=True)
    ocsvm = OneClassSVM(kernel="rbf", nu=ocsvm_nu, gamma=gamma).fit(_prep(q_fit, k_fit))
 
    # punteggio = decision_function: > 0 dentro il confine (normale), < 0 fuori (anomalo); piu' basso = piu' anomalo
    score_fit = ocsvm.decision_function(_prep(q_fit, k_fit))
    score_normal = ocsvm.decision_function(_prep(q_normal, k_normal))
    score_anom = ocsvm.decision_function(_prep(q_anom, k_anom))
    print(f"OK -> punteggi calcolati: {len(score_normal)} normali, {len(score_anom)} anomale", "\n", flush=True)
 
    # calcolo risultati, y=label -> 0=
    scores = np.concatenate([score_normal, score_anom])
    label = np.concatenate([np.zeros(len(score_normal)), np.ones(len(score_anom))])
 
    ranks = pd.Series(scores).rank().values
    auc_high = float((ranks[len(score_normal):].sum() - len(score_anom) * (len(score_anom) + 1) / 2) / (len(score_anom) * len(score_normal)))
    auc_low = 1.0 - auc_high        # anomalia = punteggio basso (fuori dal confine)
 
    tests = {
        "ocsvm score < 0": scores < 0,      # confine dell'OC-SVM
    }
 
    tests_metrics = {}
    for key, value in tests.items():
        tp = int((value & (label == 1)).sum())
        fp = int((value & (label == 0)).sum())
        fn = int((~value & (label == 1)).sum())
        tn = int((~value & (label == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        rec = tp / (tp + fn)
 
        tests_metrics[key] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp),
            "REC": rec, "PREC": prec,
            "F1": (2 * prec * rec / (prec + rec)) if (tp > 0) else 0.0,
        }
 
    results = {
        "job_name": job_name,
        "dataset_anomalies": dataset_anomalies,
        "score_type": "ocsvm_decision_function (valore basso = anomalia), vettore [q ottico; k SAR]",
        "ocsvm": {"nu": ocsvm_nu, "gamma": gamma, "standardize": standardize, "dim": int(z_fit.shape[1]),
                  "frac_outside_fit": float((score_fit < 0).mean())},
        "n_fit": len(score_fit), "n_normali": len(score_normal), "n_anomale": len(score_anom),
        "score_normal": {"mean": float(score_normal.mean()), "std": float(score_normal.std())},
        "score_anom": {"mean": float(score_anom.mean()), "std": float(score_anom.std())},
        "auroc_ocsvm_low_anom": auc_low,
        "rules": tests_metrics,
    }
 
    # risultati
 
    out_dir = f"/data/anomaly_{job_name}"
    os.makedirs(out_dir, exist_ok=True)
 
    pd.DataFrame({
        "ID": normal_test_ids + anom_ids,
        "label": [0] * len(score_normal) + [1] * len(score_anom),
        "score": np.concatenate([score_normal, score_anom]),
    }).to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    with open(os.path.join(out_dir, "anomaly_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
 
    # embedding grezzi (q = ottico, k = SAR): permettono di provare altri punteggi (kNN, sonda lineare, altri nu/gamma) senza rifare l'estrazione
    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        fit_ids=np.array(normal_fit_ids), fit_q=q_fit, fit_k=k_fit,
        normal_ids=np.array(normal_test_ids), normal_q=q_normal, normal_k=k_normal,
        anom_ids=np.array(anom_ids), anom_q=q_anom, anom_k=k_anom,
    )
    print(json.dumps(results, indent=2), flush=True)
 
    try:
        shutil.make_archive(out_dir, 'zip', out_dir)
        project_work.log_artifact(name=f"anomaly_results_{job_name}", source=f"{out_dir}.zip", kind='artifact')
        print("OK -> risultati salvati", flush=True)
    except Exception as e:
        print(f"EXC -> upload risultati: {e}", flush=True)
 
    return "TERMINATO -> anomaly detection (OC-SVM)"





# Funzioni di analisi con prefisso _ae_: src e' un unico namespace condiviso tra gli handler, il prefisso evita collisioni.


# ----------------------------------------------------------------------------------------------
# metriche
# ----------------------------------------------------------------------------------------------
def _ae_auroc(score_normal, score_anom):
    """AUROC con 'punteggio alto = normale': P(normale > anomala), 0.5 sui pareggi (Mann-Whitney da ranghi)."""
    n1, n2 = len(score_normal), len(score_anom)
    r = rankdata(np.concatenate([score_normal, score_anom]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def _ae_auroc_se(a, n1, n2):
    """Errore standard dell'AUROC (Hanley-McNeil, 1982) per n1 normali e n2 anomale."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2)))


def _ae_auroc_se_null(n1, n2):
    """Errore standard dell'AUROC se non c'e' nessun segnale (ipotesi nulla)."""
    return float(np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2)))


# ----------------------------------------------------------------------------------------------
# dati
# ----------------------------------------------------------------------------------------------
def _ae_load_embeddings(npz_path):
    with np.load(npz_path) as d:
        e = {k: d[k] for k in d.files}
    for part in ("fit", "normal", "anom"):
        for s in ("q", "k"):
            a = np.asarray(e[f"{part}_{s}"], dtype=np.float64)
            e[f"{part}_{s}"] = a.reshape(len(a), -1)
        n = len(e[f"{part}_q"])
        if len(e[f"{part}_k"]) != n or len(e[f"{part}_ids"]) != n:
            raise ValueError(f"'{part}': q, k e ids hanno lunghezze diverse "
                             f"({n}, {len(e[f'{part}_k'])}, {len(e[f'{part}_ids'])})")
    if e["fit_q"].shape[1] != e["normal_q"].shape[1] or e["fit_q"].shape[1] != e["anom_q"].shape[1]:
        raise ValueError("dimensione di q diversa tra fit, normal e anom")
    return e


def _ae_inputs(e):
    """{nome: (fit, normal, anom)} per gli ingressi q (ottico), k (SAR) e [q;k] (concatenazione)."""
    out = {}
    for name in ("q", "k"):
        out[name] = tuple(e[f"{part}_{name}"] for part in ("fit", "normal", "anom"))
    out["[q;k]"] = tuple(np.hstack([e[f"{part}_q"], e[f"{part}_k"]]) for part in ("fit", "normal", "anom"))
    return out


# ----------------------------------------------------------------------------------------------
# punteggi di normalita'
# ----------------------------------------------------------------------------------------------
def _ae_cosine_qk(q, k):
    """Similarita' coseno tra q e k della stessa serie (il punteggio usato finora)."""
    nq, nk = np.linalg.norm(q, axis=1), np.linalg.norm(k, axis=1)
    return (q * k).sum(axis=1) / (nq * nk + 1e-12)


def _ae_ocsvm_normality(X_fit, X_n, X_a, nu, gamma_mult=1.0, standardize=True):
    """OC-SVM RBF; gamma = gamma_mult x 'scale' di sklearn (1 / (n_features * var(X)) sui dati standardizzati)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    if standardize:
        sc = StandardScaler().fit(X_fit)
        X_fit, X_n, X_a = sc.transform(X_fit), sc.transform(X_n), sc.transform(X_a)
    gamma = "scale" if gamma_mult == 1.0 else gamma_mult / (X_fit.shape[1] * X_fit.var())
    m = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(X_fit)
    return m.decision_function(X_n), m.decision_function(X_a)


def _ae_knn_cosine_normality(X_fit, X_n, X_a, n_neighbors, standardize=True):
    """Media delle n_neighbors similarita' coseno piu' alte verso le serie di fit."""
    if standardize:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X_fit)
        X_fit, X_n, X_a = sc.transform(X_fit), sc.transform(X_n), sc.transform(X_a)

    def unit(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    F = unit(X_fit)

    def score(Y):
        s = unit(Y) @ F.T
        return np.partition(s, -n_neighbors, axis=1)[:, -n_neighbors:].mean(axis=1)

    return score(X_n), score(X_a)


def _ae_otsu_threshold_1d(x, bins=64):
    """Soglia di Otsu su un istogramma a `bins` classi: massimizza la varianza tra le due classi.
    Restituisce il bordo tra le due classi (= centro della classe scelta da skimage + mezza larghezza)."""
    x = np.asarray(x, dtype=np.float64)
    if x.max() == x.min():
        return float(x[0])
    counts, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w1 = np.cumsum(counts)
    w2 = np.cumsum(counts[::-1])[::-1]
    m1 = np.cumsum(counts * centers) / w1
    m2 = (np.cumsum((counts * centers)[::-1]) / w2[::-1])[::-1]
    var_between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(edges[int(np.argmax(var_between)) + 1])


def _ae_thresholds(X, method="otsu", bins=64):
    """Una soglia per dimensione, calcolata su X (N, D)."""
    if method == "median":
        return np.median(X, axis=0)
    if method == "otsu":
        return np.array([_ae_otsu_threshold_1d(X[:, j], bins) for j in range(X.shape[1])])
    raise ValueError(f"metodo soglie sconosciuto: {method}")


def _ae_hamming_counts(B1, B2):
    """(n1, n2): numero di bit diversi tra ogni riga di B1 e ogni riga di B2 (codici 0/1)."""
    a, b = B1.astype(np.float32), B2.astype(np.float32)
    return a @ (1.0 - b).T + (1.0 - a) @ b.T


def _ae_knn_hamming_normality(B_fit, B_n, B_a, n_neighbors):
    """Meno la frazione media di bit diversi dai n_neighbors codici di fit piu' vicini."""
    D = B_fit.shape[1]

    def score(B):
        h = _ae_hamming_counts(B, B_fit)
        return -np.partition(h, n_neighbors - 1, axis=1)[:, :n_neighbors].mean(axis=1) / D

    return score(B_n), score(B_a)


def _ae_bit_balance(B):
    """Quanto sono bilanciati i bit sul fit: p_min_medio = media di min(p, 1-p) (0.5 = tutti bilanciati)."""
    p = B.mean(axis=0)
    return {"dim": int(B.shape[1]),
            "p_min_medio": float(np.minimum(p, 1 - p).mean()),
            "bit_sbilanciati(<10% o >90%)": int(((p < 0.1) | (p > 0.9)).sum())}


# ----------------------------------------------------------------------------------------------
# flood per serie (per la stratificazione)
# ----------------------------------------------------------------------------------------------
def _ae_flood_px_from_masks(mask_dir, flood_class=2, out_csv=None):
    """Conta i pixel classe `flood_class` di ogni maschera nella cartella MASK/ (partNN.zip di .npy)."""
    rows = []
    for n in sorted(os.listdir(mask_dir)):
        if n.startswith("part") and n.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(mask_dir, n)) as z:
                for f in z.namelist():
                    if f.endswith(".npy"):
                        m = np.load(io.BytesIO(z.read(f)))
                        rows.append((f[:-4], int((m == flood_class).sum()), int(m.size)))
    if not rows:
        raise RuntimeError(f"nessun partNN.zip con .npy in {mask_dir}")
    df = pd.DataFrame(rows, columns=["ID", "flood_px", "total_px"])
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def _ae_stratify(scores, anom_ids, flood_df, labels, n_bins=4, min_group=30):
    """Per ogni metodo in `labels`: correlazione di Spearman tra normalita' e frazione di flood (tra le anomale)
    e AUROC contro TUTTE le normali per gruppi di anomale con frazione di flood crescente.
    Se il segnale e' flood: rho < 0 e AUROC che cresce con la frazione. Se e' dominio: rho ~ 0 e AUROC piatto."""
    f = flood_df.drop_duplicates("ID").set_index("ID")
    total = f["total_px"] if "total_px" in f.columns else pd.Series(65536, index=f.index)
    in_csv = np.array([i in f.index for i in anom_ids])
    if in_csv.sum() == 0:
        raise ValueError("nessun ID delle anomale trovato nel CSV dei flood")
    if (~in_csv).any():
        print(f"ATTENZIONE -> {int((~in_csv).sum())} anomale non nel CSV dei flood, escluse dalla stratificazione", flush=True)
    ids = [i for i, ok in zip(anom_ids, in_csv) if ok]
    frac = np.array([f.loc[i, "flood_px"] / total.loc[i] for i in ids])

    groups = []                                              # (nome, maschera booleana sulle anomale con ID nel CSV)
    zero = frac == 0
    if zero.sum() >= min_group:
        groups.append(("senza flood", zero))
    pos = frac[~zero]
    if len(pos) >= min_group:
        edges = np.unique(np.quantile(pos, np.linspace(0, 1, n_bins + 1)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = (~zero) & (frac >= lo) & ((frac <= hi) if hi == edges[-1] else (frac < hi))
            if g.sum() >= min_group:
                groups.append((f"{100 * lo:.1f}-{100 * hi:.1f}%", g))

    rows = []
    for lab in labels:
        sn, sa = scores[lab]
        sa = sa[in_csv]
        rho, p = spearmanr(sa, frac)
        row = {"metodo": lab, "rho(norm., frazione flood)": round(float(rho), 3), "p": float(p)}
        for name, g in groups:
            a = _ae_auroc(sn, sa[g])
            row[f"AUROC {name} (n={int(g.sum())})"] = f"{a:.3f}±{_ae_auroc_se(a, len(sn), int(g.sum())):.3f}"
        rows.append(row)
    return pd.DataFrame(rows), groups


def _ae_flood_probe(e, flood_df, n_splits=5, n_neighbors=20, seed=0):
    """Quanta informazione sul flood c'e' nei vettori, usando SOLO le anomale (nessuna normale: niente confusione
    di dominio). Si prevede la frazione di flood da q, da k e da [q;k] con due modelli (regressione lineare Ridge e
    kNN, entrambi su feature standardizzate) in cross-validation a gruppi: il gruppo e' la parte dell'ID prima del
    primo '_' (di solito l'evento), cosi' le serie dello stesso evento restano nello stesso fold. Sulle predizioni
    fuori campione si misurano la correlazione di Spearman (rho) e l'R2 con la frazione vera di flood.
    rho ~ 0: il vettore non contiene il flood; rho vicino a 1: lo contiene. Restituisce (DataFrame, descrizione CV)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    f = flood_df.drop_duplicates("ID").set_index("ID")
    total = f["total_px"] if "total_px" in f.columns else pd.Series(65536, index=f.index)
    ids_all = list(e["anom_ids"])
    keep = np.array([i in f.index for i in ids_all])
    if keep.sum() < 3 * n_splits:
        raise ValueError(f"troppe poche anomale con maschera ({int(keep.sum())}) per una cross-validation a {n_splits} fold")
    if (~keep).any():
        print(f"ATTENZIONE -> {int((~keep).sum())} anomale non nel CSV dei flood, escluse dalla diagnostica", flush=True)
    ids = [i for i, k in zip(ids_all, keep) if k]
    y = np.array([f.loc[i, "flood_px"] / total.loc[i] for i in ids])
    if np.ptp(y) == 0:
        raise ValueError("frazione di flood identica per tutte le anomale: niente da prevedere")

    groups = np.array([str(i).split("_")[0] for i in ids])
    n_groups = len(set(groups))
    if n_groups >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        cv_desc = f"GroupKFold a {n_splits} fold su {n_groups} gruppi (ID prima del primo '_')"
        if n_groups > 0.9 * len(ids):
            cv_desc += (": ATTENZIONE, quasi ogni serie e' un gruppo, quindi nessuna protezione dalla fuga tra serie "
                        "dello stesso evento e il rho puo' essere ottimistico")
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_desc = (f"KFold a {n_splits} fold (solo {n_groups} gruppi nell'ID: non basta per raggruppare, "
                   f"serie dello stesso evento possono finire in fold diversi)")

    Xs = {"q": e["anom_q"][keep], "k": e["anom_k"][keep], "[q;k]": np.hstack([e["anom_q"][keep], e["anom_k"][keep]])}
    models = {
        "lineare (Ridge)": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))),
        f"locale (kNN k={n_neighbors})": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors)),
    }
    rows = []
    for inp, X in Xs.items():
        for name, make in models.items():
            p = cross_val_predict(make(), X, y, cv=cv, groups=groups if isinstance(cv, GroupKFold) else None)
            rho, _ = spearmanr(p, y)
            r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            rows.append({"ingresso": inp, "modello": name, "rho": float(rho),
                         "se_rho": float((1 - rho ** 2) / np.sqrt(len(y) - 3)), "R2": float(r2), "n": int(len(y))})
    return pd.DataFrame(rows), cv_desc


# ----------------------------------------------------------------------------------------------
# esecuzione completa
# ----------------------------------------------------------------------------------------------
def _ae_run(npz_path, flood_csv=None, nu=None, standardize=None,
        gamma_mults=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), nu_list=(0.01, 0.05, 0.2),
        n_neighbors=(5, 20), otsu_bins=64, strat_top=3, n_bins=4,
        out_dir="analisi_embeddings", plot=True, show=False):
    os.makedirs(out_dir, exist_ok=True)
    e = _ae_load_embeddings(npz_path)
    n_fit, n_n, n_a = len(e["fit_q"]), len(e["normal_q"]), len(e["anom_q"])
    D = e["fit_q"].shape[1]
    print(f"fit {n_fit} | normali {n_n} | anomale {n_a} | dim q = dim k = {D} | "
          f"SE sotto ipotesi nulla = {_ae_auroc_se_null(n_n, n_a):.4f}", flush=True)

    # parametri del job, se anomaly_results.json e' accanto all'npz: nu/standardize di default e controllo di riproduzione
    job = None
    jpath = os.path.join(os.path.dirname(os.path.abspath(npz_path)), "anomaly_results.json")
    if os.path.exists(jpath):
        with open(jpath) as f:
            job = json.load(f)
    if nu is None:
        nu = job["ocsvm"]["nu"] if job and "ocsvm" in job else 0.05
    if standardize is None:
        standardize = job["ocsvm"]["standardize"] if job and "ocsvm" in job else True
    print(f"OC-SVM di riferimento: nu={nu}, standardize={standardize}"
          f"{' (dal job)' if job else ''}", flush=True)

    X = _ae_inputs(e)
    rows, scores = [], {}
    n0 = _ae_auroc_se_null(n_n, n_a)

    def add(family, inp, params, sn, sa):
        a = _ae_auroc(sn, sa)
        se = _ae_auroc_se(a, len(sn), len(sa))
        label = f"{family} | {inp} | {params}"
        scores[label] = (np.asarray(sn), np.asarray(sa))
        rows.append({"famiglia": family, "ingresso": inp, "parametri": params,
                     "auroc": a, "se": se, "sigma_dal_caso": (a - 0.5) / n0})
        return label

    # 1) baseline: coseno tra q e k
    add("coseno", "q,k", "-", _ae_cosine_qk(e["normal_q"], e["normal_k"]), _ae_cosine_qk(e["anom_q"], e["anom_k"]))

    # 2) OC-SVM: ablazione degli ingressi x sensibilita' a gamma, poi sensibilita' a nu
    print("OC-SVM ...", flush=True)
    for inp, (xf, xn, xa) in X.items():
        for gm in gamma_mults:
            sn, sa = _ae_ocsvm_normality(xf, xn, xa, nu, gm, standardize)
            add("OC-SVM", inp, f"gamma=scale x{gm} | nu={nu}", sn, sa)
    xf, xn, xa = X["[q;k]"]
    for nv in nu_list:
        if nv == nu:
            continue
        sn, sa = _ae_ocsvm_normality(xf, xn, xa, nv, 1.0, standardize)
        add("OC-SVM", "[q;k]", f"gamma=scale x1.0 | nu={nv}", sn, sa)

    # 3) kNN coseno verso le serie di fit
    print("kNN coseno ...", flush=True)
    for inp, (xf, xn, xa) in X.items():
        for nn in n_neighbors:
            sn, sa = _ae_knn_cosine_normality(xf, xn, xa, nn, standardize)
            add("kNN coseno", inp, f"k={nn}", sn, sa)

    # 4) binarizzazione per dimensione (soglie calcolate sul fit) + Hamming
    print("Binarizzazione + Hamming ...", flush=True)
    balance = {}
    for method in ("otsu", "median"):
        tag = f"soglie {method}" + (f" ({otsu_bins} classi)" if method == "otsu" else "")
        for inp, (xf, xn, xa) in X.items():
            thr = _ae_thresholds(xf, method, otsu_bins)
            Bf, Bn, Ba = (xf > thr).astype(np.uint8), (xn > thr).astype(np.uint8), (xa > thr).astype(np.uint8)
            balance[f"{method} | {inp}"] = _ae_bit_balance(Bf)
            for nn in n_neighbors:
                sn, sa = _ae_knn_hamming_normality(Bf, Bn, Ba, nn)
                add("kNN Hamming", inp, f"{tag} | k={nn}", sn, sa)
        # Hamming tra i codici dei due sensori della stessa serie (analogo binario del coseno)
        thr_q, thr_k = _ae_thresholds(e["fit_q"], method, otsu_bins), _ae_thresholds(e["fit_k"], method, otsu_bins)
        add("Hamming q-k", "q,k", f"{tag} | soglie separate",
            -((e["normal_q"] > thr_q) != (e["normal_k"] > thr_k)).mean(axis=1),
            -((e["anom_q"] > thr_q) != (e["anom_k"] > thr_k)).mean(axis=1))
        thr_s = _ae_thresholds(np.vstack([e["fit_q"], e["fit_k"]]), method, otsu_bins)      # stessa soglia per i due sensori
        add("Hamming q-k", "q,k", f"{tag} | soglia condivisa",
            -((e["normal_q"] > thr_s) != (e["normal_k"] > thr_s)).mean(axis=1),
            -((e["anom_q"] > thr_s) != (e["anom_k"] > thr_s)).mean(axis=1))

    res = pd.DataFrame(rows)

    # ------------------------------------------------------------------ stampa
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 80)
    print("\n=== AUROC (alto = distingue; 0.5 = caso) ===")
    base = res[res.famiglia == "coseno"].iloc[0]
    print(f"coseno q,k: {base.auroc:.3f} ± {base.se:.3f}\n")

    oc = res[res.famiglia == "OC-SVM"].copy()
    oc["gm"] = oc.parametri.str.extract(r"x([\d.]+)")[0].astype(float)
    oc["nu_"] = oc.parametri.str.extract(r"nu=([\d.]+)")[0].astype(float)
    piv = oc[np.isclose(oc.nu_, nu)].pivot(index="ingresso", columns="gm", values="auroc").reindex(["q", "k", "[q;k]"])
    print(f"OC-SVM, nu={nu}: ingresso x moltiplicatore di gamma (1.0 = 'scale', come il job)")
    print(piv.round(3).to_string(), "\n")
    print("OC-SVM [q;k], sensibilita' a nu (gamma=scale):")
    print(oc[(oc.ingresso == "[q;k]") & (oc.gm == 1.0)].sort_values("nu_")[["nu_", "auroc", "se"]]
          .rename(columns={"nu_": "nu"}).round(3).to_string(index=False), "\n")

    for fam in ("kNN coseno", "kNN Hamming", "Hamming q-k"):
        print(f"{fam}:")
        print(res[res.famiglia == fam][["ingresso", "parametri", "auroc", "se"]].round(3).to_string(index=False), "\n")

    print("Bilanciamento dei bit sul fit (p_min_medio 0.5 = perfetto; bit sbilanciati = dimensioni quasi costanti):")
    bal = pd.DataFrame(balance).T
    bal[["dim", "bit_sbilanciati(<10% o >90%)"]] = bal[["dim", "bit_sbilanciati(<10% o >90%)"]].astype(int)
    print(bal.round(3).to_string(), "\n")

    if job and "auroc_ocsvm_low_anom" in job and job.get("ocsvm", {}).get("gamma") == "scale":
        mine = res[(res.famiglia == "OC-SVM") & (res.ingresso == "[q;k]") &
                   (res.parametri == f"gamma=scale x1.0 | nu={nu}")].auroc.iloc[0]
        print(f"Controllo di riproduzione: AUROC job = {job['auroc_ocsvm_low_anom']:.4f} | qui = {mine:.4f} "
              f"| differenza = {abs(job['auroc_ocsvm_low_anom'] - mine):.1e}\n")

    # ------------------------------------------------------------------ stratificazione per frazione di flood
    strat = None
    if flood_csv:
        flood_df = pd.read_csv(flood_csv)
        top = res[res.famiglia != "coseno"].sort_values("auroc", ascending=False).head(strat_top)
        labels = [f"coseno | q,k | -",
                  f"OC-SVM | [q;k] | gamma=scale x1.0 | nu={nu}"] + \
                 [f"{r.famiglia} | {r.ingresso} | {r.parametri}" for r in top.itertuples()]
        labels = list(dict.fromkeys(labels))                                    # senza duplicati, ordine mantenuto
        strat, groups = _ae_stratify(scores, list(e["anom_ids"]), flood_df, labels, n_bins=n_bins)
        print("=== Per frazione di flood delle anomale (AUROC contro tutte le normali) ===")
        print("Se il segnale e' flood: rho < 0 e AUROC che sale con la frazione. Se e' dominio: rho ~ 0 e AUROC piatto.")
        print(strat.to_string(index=False), "\n")
        strat.to_csv(os.path.join(out_dir, "stratificazione_flood.csv"), index=False)

        # quanta informazione sul flood c'e' nei vettori (solo anomale, senza confusione di dominio)
        probe, cv_desc = _ae_flood_probe(e, flood_df)
        print("=== Quanta informazione sul flood c'e' nei vettori (solo anomale, nessuna normale) ===")
        print(f"Si prevede la frazione di flood dal vettore, cross-validation: {cv_desc}.")
        print("rho = correlazione di Spearman tra frazione prevista e vera, fuori campione: ~0 = il vettore non contiene il flood, "
              "vicino a 1 = lo contiene. R2 puo' essere negativo (peggio di prevedere la media).")
        print(probe.round(3).to_string(index=False))
        best_unsup = float(strat["rho(norm., frazione flood)"].abs().max())
        print(f"Per confronto: il miglior |rho| dei punteggi non supervisionati mostrati sopra e' {best_unsup:.3f}. "
              f"Se il rho della regressione e' molto piu' alto, l'informazione c'e' e il rilevatore non la sfrutta. "
              f"Attenzione: nuvole e altre caratteristiche della scena correlate col flood possono contribuire.\n")
        probe.to_csv(os.path.join(out_dir, "diagnostica_flood.csv"), index=False)

    res.to_csv(os.path.join(out_dir, "auroc_metodi.csv"), index=False)

    # ------------------------------------------------------------------ grafico
    if plot:
        import matplotlib.pyplot as plt
        keep = pd.concat([res.sort_values("auroc", ascending=False).head(18),
                          res[res.famiglia == "coseno"]]).drop_duplicates(["famiglia", "ingresso", "parametri"])
        keep = keep.sort_values("auroc")
        ncol = 2 if strat is not None else 1
        fig, axes = plt.subplots(1, ncol, figsize=(7 + 7 * (ncol - 1), 0.32 * len(keep) + 2), squeeze=False)
        ax = axes[0, 0]
        names = [f"{r.famiglia} | {r.ingresso} | {r.parametri}" for r in keep.itertuples()]
        ax.barh(range(len(keep)), keep.auroc - 0.5, left=0.5, xerr=keep.se, color="#4C78A8")
        ax.set_yticks(range(len(keep)))
        ax.set_yticklabels(names, fontsize=7)
        ax.axvline(0.5, color="k", lw=1)
        ax.set_xlim(0.45, max(0.7, float(keep.auroc.max()) + 0.03))
        ax.set_xlabel("AUROC (normali vs anomale)")
        ax.set_title("Metodi (migliori + coseno)")
        if strat is not None:
            ax = axes[0, 1]
            cols = [c for c in strat.columns if c.startswith("AUROC ")]
            for _, r in strat.iterrows():
                vals = [float(r[c].split("±")[0]) for c in cols]
                ax.plot(range(len(cols)), vals, marker="o", label=str(r["metodo"])[:48])
            ax.axhline(0.5, color="k", lw=1)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([c.replace("AUROC ", "").replace(" (n=", "\n(n=") for c in cols], fontsize=8)
            ax.set_ylabel("AUROC")
            ax.set_xlabel("frazione di flood delle anomale (crescente)")
            ax.set_title("AUROC per frazione di flood")
            ax.legend(fontsize=6)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "analisi_embeddings.png"), dpi=130, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    print(f"OK -> risultati in {os.path.abspath(out_dir)}", flush=True)
    return res, scores


class _AeTee:
    """Scrive la stessa uscita sul log del job e su report.txt."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@handler()
def analyze_anomalies(
    results_job: str = "net_v3_ocsvm",      # job OC-SVM da analizzare: legge l'artifact anomaly_results_{results_job}
    dataset_masks: str = "",                # artifact con MASK/ (partNN.zip di .npy); "" = dataset_anomalies scritto nel JSON del job
    flood_class: int = 2,                   # valore della classe flood nella maschera
    nu: float = 0.0,                        # nu dell'OC-SVM di riferimento; 0 = quello del job
    otsu_bins: int = 64,                    # classi dell'istogramma per le soglie di Otsu
    n_flood_bins: int = 4,                  # gruppi di anomale per frazione di flood (piu' un gruppo "senza flood" se ce ne sono)
    job_name: str = "analisi_net_v3_ocsvm",
    save_dir: str = "/data/analisi_embeddings",
):
    """
    Analisi offline degli embedding (q ottico, k SAR) salvati da train_anomaly_detection_ocsvm, senza GPU e senza MoCo:
      - AUROC di coseno, OC-SVM (q / k / [q;k], sensibilita' a gamma e nu), kNN coseno, kNN Hamming e Hamming q-k
        con soglie per dimensione di Otsu o mediana (calcolate sulle normali di fit);
      - AUROC e correlazione di Spearman per frazione di flood delle anomale (pixel classe `flood_class` della maschera):
        se il segnale e' flood l'AUROC sale con la frazione, se e' dominio resta piatto;
      - diagnostica sul solo insieme delle anomale (nessuna normale, quindi senza confusione di dominio): quanta
        informazione sulla frazione di flood c'e' in q, in k e in [q;k] (regressione lineare e kNN, cross-validation
        a gruppi per evento). Il flood e le maschere servono solo a misurare, non entrano nel rilevatore.
    Il risultato esce come artifact analisi_embeddings_{job_name} (CSV, grafico, report.txt, flood_px.csv).
    """
    import matplotlib
    matplotlib.use("Agg")

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    work_dir = save_dir + "_" + job_name
    shutil.rmtree(work_dir, ignore_errors=True)
    in_dir = os.path.join(work_dir, "in")
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1) risultati del job OC-SVM: embeddings.npz + anomaly_results.json
    print(f"Download: anomaly_results_{results_job}", flush=True)
    zip_in = project_work.get_artifact(f"anomaly_results_{results_job}").download(
        os.path.join(in_dir, "anomaly_results.zip"), overwrite=True)
    res_dir = os.path.join(in_dir, "results")
    with zipfile.ZipFile(zip_in) as z:
        z.extractall(res_dir)
    npz_path = os.path.join(res_dir, "embeddings.npz")
    json_path = os.path.join(res_dir, "anomaly_results.json")
    if not os.path.exists(npz_path):
        raise RuntimeError(f"embeddings.npz non trovato in anomaly_results_{results_job} (contenuto: {os.listdir(res_dir)}): "
                           f"serve un job eseguito con la versione di train_anomaly_detection_ocsvm che lo salva")
    with open(json_path) as f:
        job = json.load(f)
    print(f"OK -> job {job.get('job_name')} | n_fit {job.get('n_fit')} | n_normali {job.get('n_normali')} | "
          f"n_anomale {job.get('n_anomale')} | AUROC job {job.get('auroc_ocsvm_low_anom')}", flush=True)

    # 2) maschere delle anomale -> pixel flood per serie
    masks_name = dataset_masks or job.get("dataset_anomalies", "")
    if not masks_name:
        raise RuntimeError("dataset_masks vuoto e dataset_anomalies assente nel JSON del job: passa dataset_masks")
    print(f"Download: {masks_name} (serve solo MASK/)", flush=True)
    dataset_path = project_data.get_artifact(masks_name).download(os.path.join(work_dir, "dataset"))
    mask_dir = os.path.join(dataset_path, "MASK")
    if not os.path.isdir(mask_dir):
        raise RuntimeError(f"cartella MASK/ non trovata in {masks_name} (contenuto: {os.listdir(dataset_path)}): "
                           f"serve un artifact costruito con la nuova crop_norm_artifact_anomalies, passalo in dataset_masks")
    flood_csv = os.path.join(out_dir, "flood_px.csv")
    flood_df = _ae_flood_px_from_masks(mask_dir, flood_class=flood_class, out_csv=flood_csv)
    print(f"OK -> {len(flood_df)} maschere lette da {masks_name}/MASK", flush=True)

    with np.load(npz_path) as d:
        anom_ids = set(d["anom_ids"].tolist())
    n_cov = len(anom_ids & set(flood_df.ID))
    print(f"OK -> maschera trovata per {n_cov}/{len(anom_ids)} serie anomale", flush=True)
    if n_cov == 0:
        raise RuntimeError("nessuna serie anomala ha una maschera: dataset_masks non corrisponde alle serie del job")

    # 3) analisi (stampa sul log e su report.txt)
    with open(os.path.join(out_dir, "report.txt"), "w") as rep:
        with contextlib.redirect_stdout(_AeTee(sys.stdout, rep)):
            res, scores = _ae_run(npz_path, flood_csv=flood_csv, nu=(nu if nu > 0 else None), standardize=None,
                                  otsu_bins=otsu_bins, n_bins=n_flood_bins, out_dir=out_dir, plot=True, show=False)

    best = res.sort_values("auroc", ascending=False).iloc[0]
    with open(os.path.join(out_dir, "riassunto.json"), "w") as f:
        json.dump({"results_job": results_job, "dataset_masks": masks_name, "n_maschere_anomale": int(n_cov),
                   "auroc_job": job.get("auroc_ocsvm_low_anom"),
                   "migliore": {"famiglia": best.famiglia, "ingresso": best.ingresso, "parametri": best.parametri,
                                "auroc": float(best.auroc), "se": float(best.se)}}, f, indent=2)

    zip_base = work_dir.rstrip("/") + "_out"
    shutil.make_archive(zip_base, "zip", out_dir)
    project_work.log_artifact(
        name=f"analisi_embeddings_{job_name}",
        kind="artifact",
        source=f"{zip_base}.zip",
    )
    print(f"OK -> artifact analisi_embeddings_{job_name}: {sorted(os.listdir(out_dir))}", flush=True)

    return "TERMINATO"    