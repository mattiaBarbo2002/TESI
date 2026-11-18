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
import matplotlib.pyplot as plt
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
from sklearn.covariance import LedoitWolf, ledoit_wolf_shrinkage

from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_2d
from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_attn

from floodlib.loader import MoCo2encodersLoader

from floodlib.builder import MoCo2encoders_2d
from floodlib.builder import MoCo2encoders_attn
from floodlib.builder import MoCo2encoders_Bot
from floodlib.builder import MoCo2encoders_ms
from floodlib.builder import MoCo2encoders_ms_pre

from floodlib.mocoDense import MoCo2encoders_dense 

matplotlib.use("Agg") 






@handler()
def ad_ocsvm(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
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
    
    do_encoder: bool = True,                        # latente autoencoder + pooling di moco
    do_pre_proj: bool = True,                       # dopo conv + pooling, prima del proiettore
    do_proj_raw: bool = True,                       # dopo il proiettore, prima della normalizzazione
    do_post_proj: bool = True,                      # dopo la normalizzazione (come contrastive_loss)
    
    pca_encoder: float | int | None = None,
    pca_pre_proj: float | int | None = None,         # pre_proj ha molte dimensioni: serve ridurre
    pca_proj_raw: float | int | None = None,
    pca_post_proj: float | int | None = None,
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    # punti selezionati e componenti pca per punto
    punti = [p for p, do in (("encoder", do_encoder), ("pre_proj", do_pre_proj),
                             ("proj_raw", do_proj_raw), ("post_proj", do_post_proj)) if do]
    pca_punto = {"encoder": pca_encoder, "pre_proj": pca_pre_proj,
                 "proj_raw": pca_proj_raw, "post_proj": pca_post_proj}
    if not punti:
        raise ValueError("nessun punto selezionato: attiva almeno un do_*")
    print(f"punti: {', '.join(punti)}", "\n", flush=True)

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

    # embedding per punto -> E[punto] = {"fit": (q, k), "thr": (q, k), "normal": (q, k), "anom": (q, k)}
    E = {punto: {} for punto in punti}

    if old_vectors:
        print(f"Upload embedding: anomaly_results_ocsvm_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_ocsvm_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            old_results = json.loads(z.read("summary.json"))                      
            for punto in punti:
                z.extract(f"{punto}/embeddings.npz", "/data/old_vectors")

        for key, value in (("moco_version", moco_version), ("dataset_anomalies", dataset_anomalies)):
            if old_results.get(key) != value:
                raise ValueError(f"old_vectors={old_vectors} ha {key}={old_results.get(key)}, questo job ha {value}")

        for punto in punti:
            old = np.load(f"/data/old_vectors/{punto}/embeddings.npz")
            lookup = {}
            for g in ("fit", "thr", "normal", "anom"):
                for i, ID in enumerate(old[f"{g}_ids"]):
                    lookup[str(ID)] = (old[f"{g}_q"][i], old[f"{g}_k"][i])

            def take(ids, nome):
                missing = [ID for ID in ids if ID not in lookup]
                if missing:
                    raise ValueError(f"{len(missing)} serie {nome} non presenti in {old_vectors} (es. {missing[:3]})")
                return np.stack([lookup[ID][0] for ID in ids]), np.stack([lookup[ID][1] for ID in ids])

            E[punto]["fit"] = take(normal_fit_ids, "fit")
            E[punto]["thr"] = take(normal_thr_ids, "thr")
            E[punto]["normal"] = take(normal_test_ids, "normali test")
            E[punto]["anom"] = take(anom_ids, "anomale")
            print(f"OK -> {punto}: embedding caricati", flush=True)
        print(flush=True)

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
                ).to(device)

                w_path = project_work.get_artifact(f"moco_2D_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)

            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("OK -> modello MoCo caricato",  "\n", flush=True)

        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
            raise                                                                   

        # ritorna {punto: (q, k)}, q = ottico, k = sar
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            out = {punto: ([], []) for punto in punti}
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)

                    # encoder: latente autoencoder + pooling di moco
                    zq = model.encoder_q(im_q)
                    zk = model.encoder_k(im_k)
                    fq = {"encoder": model.pool_q(zq).flatten(1)}
                    fk = {"encoder": model.pool_k(zk).flatten(1)}

                    # pre_proj: conv + pooling (come contrastive_loss)
                    if model.simple_proj:
                        hq = model.pool_q(zq).flatten(1)
                        hk = model.pool_k(zk).flatten(1)
                    else:
                        hq = model.pool_q(model.conv_q(zq)).flatten(1)
                        hk = model.pool_k(model.conv_k(zk)).flatten(1)
                    fq["pre_proj"], fk["pre_proj"] = hq, hk

                    # proj_raw: proiettore senza normalizzazione
                    pq = model.proj_q(hq)
                    pk = model.proj_k(hk)
                    fq["proj_raw"], fk["proj_raw"] = pq, pk

                    # post_proj: proiettore + normalizzazione (uguale a contrastive_loss)
                    fq["post_proj"] = nn.functional.normalize(pq, dim=1)
                    fk["post_proj"] = nn.functional.normalize(pk, dim=1)

                    for punto in punti:
                        out[punto][0].append(fq[punto].float().flatten(1).cpu())
                        out[punto][1].append(fk[punto].float().flatten(1).cpu())
            return {punto: (torch.cat(q).numpy(), torch.cat(k).numpy()) for punto, (q, k) in out.items()}

        def print_dim(nome, emb):
            dims = " | ".join(f"{punto} q {q.shape[1]} + k {k.shape[1]}" for punto, (q, k) in emb.items())
            print(f"OK -> {nome}: {len(next(iter(emb.values()))[0])} serie | {dims}", "\n", flush=True)

        # embedding serie di fit
        print("Embedding MoCo: serie di fit ocsvm", flush=True)
        emb = embed(normal_fit_ids)
        for punto in punti: E[punto]["fit"] = emb[punto]
        print_dim("normali fit", emb)

        # embedding serie thr
        print("Embedding MoCo: serie thr ocsvm", flush=True)
        emb = embed(normal_thr_ids)
        for punto in punti: E[punto]["thr"] = emb[punto]
        print_dim("normali thr", emb)

        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        emb = embed(normal_test_ids)
        for punto in punti: E[punto]["normal"] = emb[punto]
        print_dim("normali test", emb)

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
        emb = embed(anom_ids)
        for punto in punti: E[punto]["anom"] = emb[punto]
        print_dim("anomale", emb)

    # sottocartella per punto
    out_root = f"/data/anomaly_results_ocsvm_{job_name}"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root)
    riepilogo = {}

    # anomaly detection per punto
    for punto in punti:
        print(f"===== {punto} =====", flush=True)
        q_fit, k_fit = E[punto]["fit"]
        q_thr, k_thr = E[punto]["thr"]
        q_normal, k_normal = E[punto]["normal"]
        q_anom, k_anom = E[punto]["anom"]

        # concatenazione
        w_fit = np.concatenate([q_fit, k_fit], axis=1).astype(np.float64)
        w_thr = np.concatenate([q_thr, k_thr], axis=1).astype(np.float64)
        w_normal = np.concatenate([q_normal, k_normal], axis=1).astype(np.float64)
        w_anom = np.concatenate([q_anom, k_anom],axis=1).astype(np.float64)

        if standardize:
            scaler = StandardScaler().fit(w_fit)
            w_fit, w_thr, w_normal, w_anom = scaler.transform(w_fit), scaler.transform(w_thr), scaler.transform(w_normal), scaler.transform(w_anom)

         
        pw = PCA(n_components=pca_punto[punto], whiten=True).fit(w_fit)               
        print(f"PCA: {pw.n_components_} componenti su {w_fit.shape[1]} dimensioni", flush=True)
        z_fit = pw.transform(w_fit)
        z_thr = pw.transform(w_thr)
        z_normal = pw.transform(w_normal)
        z_anom = pw.transform(w_anom)
        

        """
        lw = LedoitWolf().fit(w_fit)
        lam, V = np.linalg.eigh(lw.covariance_)           
        W = V / np.sqrt(lam)                              
        z_fit = (w_fit - lw.location_) @ W
        z_thr = (w_thr - lw.location_) @ W
        z_normal = (w_normal - lw.location_) @ W
        z_anom = (w_anom - lw.location_) @ W
        n_componenti = w_fit.shape[1]
        shrinkage = float(lw.shrinkage_)
        print(f"Ledoit-Wolf: {n_componenti} dimensioni | shrinkage {shrinkage:.3f} | "
                f"autovalore min {lam.min():.4f} max {lam.max():.2f}", flush=True)
        """        

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
        out_dir = os.path.join(out_root, punto)                                       
        os.makedirs(out_dir)

        # punteggi
        df = pd.DataFrame({
            "ID": normal_test_ids + anom_ids,
            "label": [0] * len(score_normal) + [1] * len(score_anom),
            "score": np.concatenate([score_normal, score_anom]),
        })
        df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
        pd.DataFrame({"ID": normal_thr_ids, "score": score_thr}).to_csv(os.path.join(out_dir, "scores_thr.csv"), index=False)  


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
        print(flush=True)

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
            "feature_point": punto, "pca_components": pca_punto[punto], "n_componenti": n_componenti,   
            "ocsvm": {"nu": ocsvm_nu, "gamma": gamma, "dim": int(z_fit.shape[1]), "frac_outside_fit": frac_outside_fit},
            "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
            "auroc": float(auroc),
            "rules": rules,
        }
        with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        riepilogo[punto] = {"auroc": float(auroc), "n_componenti": n_componenti}

    
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump({"job_name": job_name, "moco_version": moco_version, "dataset_anomalies": dataset_anomalies,
                   "punti": riepilogo}, f, indent=2)

    print("===== RIEPILOGO =====", flush=True)
    for punto, r in riepilogo.items():
        print(f"{punto:9s}: auroc {r['auroc']:.4f} | pca {r['n_componenti']} componenti", flush=True)

    shutil.make_archive(out_root, "zip", out_root)
    project_work.log_artifact(name=f"anomaly_results_ocsvm_{job_name}", source=f"{out_root}.zip", kind="artifact")

    return "TERMINATO -> anomaly detection ocsvm"





@handler()
def ad_ocsvm2(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
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
    n_series_thr: int = 500,                        # usato solo con fit_on_train=False
    moco_attn: bool = False,

    do_encoder: bool = True,                        # latente autoencoder + pooling di moco
    do_pre_proj: bool = True,                       # dopo conv + pooling, prima del proiettore
    do_proj_raw: bool = True,                       # dopo il proiettore, prima della normalizzazione
    do_post_proj: bool = True,                      # dopo la normalizzazione (come contrastive_loss)

    pca_encoder: float | int | None = None,
    pca_pre_proj: float | int | None = None,
    pca_proj_raw: float | int | None = None,
    pca_post_proj: float | int | None = None,

    whitening: str = "pca",                         # "pca" | "ledoitwolf"
    fit_on_train: bool = True,                      # True: fit su tutte le serie di train di moco
    max_fit_series: int = 0,                        # 0 = tutte; altrimenti campione casuale (ocsvm lento con molte serie)
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    # punti selezionati e componenti pca per punto
    punti = [p for p, do in (("encoder", do_encoder), ("pre_proj", do_pre_proj), ("proj_raw", do_proj_raw), ("post_proj", do_post_proj)) if do]
    pca_punto = {"encoder": pca_encoder, "pre_proj": pca_pre_proj, "proj_raw": pca_proj_raw, "post_proj": pca_post_proj}

    if not punti:
        raise ValueError("nessun punto selezionato: attiva almeno un do_*")
    
    if whitening not in ("pca", "lw"):
        raise ValueError(f"whitening non valido: {whitening}")
    print(f"punti: {', '.join(punti)} | whitening: {whitening} | fit_on_train: {fit_on_train}", "\n", flush=True)

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

    normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]             # anomaly detection

    if fit_on_train:
        #fit su tutte le serie di train di moco
        if moco_attn:
            p_train = project_work.get_artifact(f"moco_attn_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
        else:
            p_train = project_work.get_artifact(f"moco_2D_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
        with open(p_train, 'r') as f:
            train_ids = json.load(f)

        normal_fit_ids = [ID for ID in train_ids if ID in normal_ids]                              # fit -> iperpiano
        if max_fit_series > 0 and len(normal_fit_ids) > max_fit_series:
            random.seed(1)
            normal_fit_ids = random.sample(normal_fit_ids, max_fit_series)

        test_set = set(normal_test_ids)
        normal_thr_ids = [ID for ID in shuffled_test_ids if ID not in test_set]                    # calcolo soglie
    else:
        normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]                        # fit -> iperpiano
        normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]          # calcolo soglie

    assert not (set(normal_fit_ids) & set(normal_thr_ids)), "serie in comune tra fit e soglia"
    assert not (set(normal_fit_ids) & set(normal_test_ids)), "serie in comune tra fit e test"
    assert not (set(normal_thr_ids) & set(normal_test_ids)), "serie in comune tra soglia e test"

    print(f"OK -> {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali", "\n", flush=True)

    # embedding per punto -> E[punto] = {"fit": (q, k), "thr": (q, k), "normal": (q, k), "anom": (q, k)}
    E = {punto: {} for punto in punti}

    if old_vectors:
        print(f"Upload embedding: anomaly_results_ocsvm_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_ocsvm_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            old_results = json.loads(z.read("summary.json"))
            for punto in punti:
                z.extract(f"{punto}/embeddings.npz", "/data/old_vectors")

        for key, value in (("moco_version", moco_version), ("dataset_anomalies", dataset_anomalies),
                           ("fit_on_train", fit_on_train)):                                         
            if old_results.get(key) != value:
                raise ValueError(f"old_vectors={old_vectors} ha {key}={old_results.get(key)}, questo job ha {value}")

        for punto in punti:
            old = np.load(f"/data/old_vectors/{punto}/embeddings.npz")
            lookup = {}
            for g in ("fit", "thr", "normal", "anom"):
                for i, ID in enumerate(old[f"{g}_ids"]):
                    lookup[str(ID)] = (old[f"{g}_q"][i], old[f"{g}_k"][i])

            def take(ids, nome):
                missing = [ID for ID in ids if ID not in lookup]
                if missing:
                    raise ValueError(f"{len(missing)} serie {nome} non presenti in {old_vectors} (es. {missing[:3]})")
                return np.stack([lookup[ID][0] for ID in ids]), np.stack([lookup[ID][1] for ID in ids])

            E[punto]["fit"] = take(normal_fit_ids, "fit")
            E[punto]["thr"] = take(normal_thr_ids, "thr")
            E[punto]["normal"] = take(normal_test_ids, "normali test")
            E[punto]["anom"] = take(anom_ids, "anomale")
            print(f"OK -> {punto}: embedding caricati", flush=True)
        print(flush=True)

    else:

        print("calcolo vettori MoCo", flush = True)

        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"

        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids)}
        print(f"Estrazione {len(wanted_normal_series)} serie normali", flush=True)
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
                ).to(device)

                w_path = project_work.get_artifact(f"moco_2D_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)

            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("OK -> modello MoCo caricato",  "\n", flush=True)

        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
            raise

        # embedding moco in tutti i punti selezionati, un solo passaggio
        # ritorna {punto: (q, k)}, q = ottico, k = sar
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            out = {punto: ([], []) for punto in punti}
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)

                    # encoder: latente autoencoder + pooling di moco
                    zq = model.encoder_q(im_q)
                    zk = model.encoder_k(im_k)
                    fq = {"encoder": model.pool_q(zq).flatten(1)}
                    fk = {"encoder": model.pool_k(zk).flatten(1)}

                    # pre_proj: conv + pooling (come contrastive_loss)
                    if model.simple_proj:
                        hq = model.pool_q(zq).flatten(1)
                        hk = model.pool_k(zk).flatten(1)
                    else:
                        hq = model.pool_q(model.conv_q(zq)).flatten(1)
                        hk = model.pool_k(model.conv_k(zk)).flatten(1)
                    fq["pre_proj"], fk["pre_proj"] = hq, hk

                    # proj_raw: proiettore senza normalizzazione
                    pq = model.proj_q(hq)
                    pk = model.proj_k(hk)
                    fq["proj_raw"], fk["proj_raw"] = pq, pk

                    # post_proj: proiettore + normalizzazione (uguale a contrastive_loss)
                    fq["post_proj"] = nn.functional.normalize(pq, dim=1)
                    fk["post_proj"] = nn.functional.normalize(pk, dim=1)

                    for punto in punti:
                        out[punto][0].append(fq[punto].float().flatten(1).cpu())
                        out[punto][1].append(fk[punto].float().flatten(1).cpu())
            return {punto: (torch.cat(q).numpy(), torch.cat(k).numpy()) for punto, (q, k) in out.items()}

        def print_dim(nome, emb):
            dims = " | ".join(f"{punto} q {q.shape[1]} + k {k.shape[1]}" for punto, (q, k) in emb.items())
            print(f"OK -> {nome}: {len(next(iter(emb.values()))[0])} serie | {dims}", "\n", flush=True)

        # embedding serie di fit
        print("Embedding MoCo: serie di fit ocsvm", flush=True)
        emb = embed(normal_fit_ids)
        for punto in punti: E[punto]["fit"] = emb[punto]
        print_dim("normali fit", emb)

        # embedding serie thr
        print("Embedding MoCo: serie thr ocsvm", flush=True)
        emb = embed(normal_thr_ids)
        for punto in punti: E[punto]["thr"] = emb[punto]
        print_dim("normali thr", emb)

        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        emb = embed(normal_test_ids)
        for punto in punti: E[punto]["normal"] = emb[punto]
        print_dim("normali test", emb)

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
        emb = embed(anom_ids)
        for punto in punti: E[punto]["anom"] = emb[punto]
        print_dim("anomale", emb)

    # cartella radice, una sottocartella per punto
    out_root = f"/data/anomaly_results_ocsvm_{job_name}"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root)
    riepilogo = {}

    # anomaly detection ripetuta per ogni punto
    for punto in punti:
        print(f"===== {punto} =====", flush=True)
        q_fit, k_fit = E[punto]["fit"]
        q_thr, k_thr = E[punto]["thr"]
        q_normal, k_normal = E[punto]["normal"]
        q_anom, k_anom = E[punto]["anom"]

        # concatenazione
        w_fit = np.concatenate([q_fit, k_fit], axis=1).astype(np.float64)
        w_thr = np.concatenate([q_thr, k_thr], axis=1).astype(np.float64)
        w_normal = np.concatenate([q_normal, k_normal], axis=1).astype(np.float64)
        w_anom = np.concatenate([q_anom, k_anom],axis=1).astype(np.float64)

        if standardize:
            scaler = StandardScaler().fit(w_fit)
            w_fit, w_thr, w_normal, w_anom = scaler.transform(w_fit), scaler.transform(w_thr), scaler.transform(w_normal), scaler.transform(w_anom)

        if whitening == "lw":
            lw = LedoitWolf().fit(w_fit)
            lam, V = np.linalg.eigh(lw.covariance_)           
            W = V / np.sqrt(lam)                              
            z_fit = (w_fit - lw.location_) @ W
            z_thr = (w_thr - lw.location_) @ W
            z_normal = (w_normal - lw.location_) @ W
            z_anom = (w_anom - lw.location_) @ W
            n_componenti = w_fit.shape[1]
            shrinkage = float(lw.shrinkage_)
            print(f"Ledoit-Wolf: {n_componenti} dimensioni | shrinkage {shrinkage:.3f} | "
                  f"autovalore min {lam.min():.4f} max {lam.max():.2f}", flush=True)
        else:
            pw = PCA(n_components=pca_punto[punto], whiten=True).fit(w_fit)
            z_fit = pw.transform(w_fit)
            z_thr = pw.transform(w_thr)
            z_normal = pw.transform(w_normal)
            z_anom = pw.transform(w_anom)
            n_componenti = int(pw.n_components_)
            shrinkage = None
            print(f"PCA: {n_componenti} componenti su {w_fit.shape[1]} dimensioni", flush=True)

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
        out_dir = os.path.join(out_root, punto)
        os.makedirs(out_dir)

        # punteggi
        df = pd.DataFrame({
            "ID": normal_test_ids + anom_ids,
            "label": [0] * len(score_normal) + [1] * len(score_anom),
            "score": np.concatenate([score_normal, score_anom]),
        })
        df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
        pd.DataFrame({"ID": normal_thr_ids, "score": score_thr}).to_csv(os.path.join(out_dir, "scores_thr.csv"), index=False)


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
        print(flush=True)

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
            "feature_point": punto, "pca_components": pca_punto[punto], "n_componenti": n_componenti,
            "whitening": whitening, "shrinkage": shrinkage, "fit_on_train": fit_on_train,                   
            "ocsvm": {"nu": ocsvm_nu, "gamma": gamma, "dim": int(z_fit.shape[1]), "frac_outside_fit": frac_outside_fit},
            "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
            "auroc": float(auroc),
            "rules": rules,
        }
        with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        riepilogo[punto] = {"auroc": float(auroc), "n_componenti": n_componenti, "shrinkage": shrinkage}

    # riepilogo di tutti i punti (serve anche per old_vectors)
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump({"job_name": job_name, "moco_version": moco_version, "dataset_anomalies": dataset_anomalies,
                   "fit_on_train": fit_on_train, "whitening": whitening,                                     
                   "punti": riepilogo}, f, indent=2)

    print("===== RIEPILOGO =====", flush=True)
    for punto, r in riepilogo.items():
        print(f"{punto:9s}: auroc {r['auroc']:.4f} | {r['n_componenti']} componenti", flush=True)

    shutil.make_archive(out_root, "zip", out_root)
    project_work.log_artifact(name=f"anomaly_results_ocsvm_{job_name}", source=f"{out_root}.zip", kind="artifact")

    return "TERMINATO -> anomaly detection ocsvm"







@handler()
def ad_ocsvm3(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
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
    n_series_thr: int = 500,                        # usato solo con fit_on_train=False
    moco_attn: bool = False,
    moco_bot: bool = False,
    moco_ms: bool = False,

    do_encoder: bool = True,                        # latente autoencoder + pooling di moco
    do_pre_proj: bool = True,                       # dopo conv + pooling, prima del proiettore
    do_proj_raw: bool = True,                       # dopo il proiettore, prima della normalizzazione
    do_post_proj: bool = True,                      # dopo la normalizzazione (come contrastive_loss)

    pca_encoder: float | int | None = None,
    pca_pre_proj: float | int | None = None,
    pca_proj_raw: float | int | None = None,
    pca_post_proj: float | int | None = None,

    whitening: str = "pca",                         # "pca" | "lw"
    fit_on_train: bool = True,                      # True: fit su tutte le serie di train di moco
    max_fit_series: int = 0,                        # 0 = tutte; altrimenti campione casuale (ocsvm lento con molte serie)

    weights_version: str = "",
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    # punti selezionati e componenti pca per punto
    punti = [p for p, do in (("encoder", do_encoder), ("pre_proj", do_pre_proj), ("proj_raw", do_proj_raw), ("post_proj", do_post_proj)) if do]
    pca_punto = {"encoder": pca_encoder, "pre_proj": pca_pre_proj, "proj_raw": pca_proj_raw, "post_proj": pca_post_proj}

    if not punti:
        raise ValueError("nessun punto selezionato: attiva almeno un do_*")
    
    if whitening not in ("pca", "lw"):
        raise ValueError(f"whitening non valido: {whitening}")
    print(f"punti: {', '.join(punti)} | whitening: {whitening} | fit_on_train: {fit_on_train}", "\n", flush=True)

    project_work = dh.get_project("floods")

    # embedding per punto -> E[punto] = {"fit": (q, k), "thr": (q, k), "normal": (q, k), "anom": (q, k)}
    E = {punto: {} for punto in punti}

    if old_vectors:
        print(f"Upload embedding: anomaly_results_ocsvm_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_ocsvm_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            old_summary = json.loads(z.read("summary.json"))
            mancanti = [punto for punto in punti if f"{punto}/embeddings.npz" not in z.namelist()]
            if mancanti:
                raise ValueError(f"punti non presenti in {old_vectors}: {mancanti}")
            for punto in punti:
                z.extract(f"{punto}/embeddings.npz", "/data/old_vectors")
                z.extract(f"{punto}/anomaly_results.json", "/data/old_vectors")

        moco_version = old_summary["moco_version"]
        dataset_anomalies = old_summary["dataset_anomalies"]
        fit_on_train = old_summary.get("fit_on_train", False)
        with open(f"/data/old_vectors/{punti[0]}/anomaly_results.json") as f:
            min_flood_area = json.load(f)["min_flood_area"]

        ids_rif = None
        for punto in punti:
            old = np.load(f"/data/old_vectors/{punto}/embeddings.npz")
            for g in ("fit", "thr", "normal", "anom"):
                E[punto][g] = (old[f"{g}_q"], old[f"{g}_k"])

            ids = {g: [str(ID) for ID in old[f"{g}_ids"]] for g in ("fit", "thr", "normal", "anom")}
            if ids_rif is None:
                ids_rif = ids
            elif ids != ids_rif:
                raise ValueError(f"il punto {punto} ha serie diverse dagli altri punti")
            print(f"OK -> {punto}: embedding caricati", flush=True)

        normal_fit_ids, normal_thr_ids = ids_rif["fit"], ids_rif["thr"]
        normal_test_ids, anom_ids = ids_rif["normal"], ids_rif["anom"]
        print(f"OK -> {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali, "
              f"{len(anom_ids)} anomale | moco {moco_version} | flood >= {min_flood_area:g}%", "\n", flush=True)

    else:
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
        elif moco_bot:
            p = project_work.get_artifact(f"moco_bot_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
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

        normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]             # anomaly detection

        if fit_on_train:
            #fit su tutte le serie di train di moco
            if moco_attn:
                p_train = project_work.get_artifact(f"moco_attn_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
            elif moco_bot:
                p_train = project_work.get_artifact(f"moco_bot_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
            else:    
                p_train = project_work.get_artifact(f"moco_2D_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
            with open(p_train, 'r') as f:
                train_ids = json.load(f)

            normal_fit_ids = [ID for ID in train_ids if ID in normal_ids]                              # fit -> iperpiano
            if max_fit_series > 0 and len(normal_fit_ids) > max_fit_series:
                random.seed(1)
                normal_fit_ids = random.sample(normal_fit_ids, max_fit_series)

            test_set = set(normal_test_ids)
            normal_thr_ids = [ID for ID in shuffled_test_ids if ID not in test_set]                    # calcolo soglie
        else:
            normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]                        # fit -> iperpiano
            normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]          # calcolo soglie

        assert not (set(normal_fit_ids) & set(normal_thr_ids)), "serie in comune tra fit e soglia"
        assert not (set(normal_fit_ids) & set(normal_test_ids)), "serie in comune tra fit e test"
        assert not (set(normal_thr_ids) & set(normal_test_ids)), "serie in comune tra soglia e test"

        print(f"OK -> {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali", "\n", flush=True)

        print("calcolo vettori MoCo", flush = True)

        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"

        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids)}
        print(f"Estrazione {len(wanted_normal_series)} serie normali", flush=True)
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)

        # caricamento pesi moco
        try:

            def scarica_pesi(nome_artifact):
                if weights_version:
                    art = dh.get_artifact(nome_artifact, project="floods", entity_id=weights_version)
                else:
                    art = project_work.get_artifact(nome_artifact)
                print(f"pesi: {nome_artifact} (versione {weights_version or 'ultima'})", flush=True)
                return art.download("/data/moco_weights.pth", overwrite=True)
            
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
                ).to(device)

                w_path = scarica_pesi(f"moco_attn_weights_{moco_version}")              

            else:
                encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
                encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)

                if moco_bot:
                    model = MoCo2encoders_Bot(
                            base_encoder_q=encoderOPT.encoder,
                            base_encoder_k=encoderSAR.encoder,
                            dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                            symmetric=False, device=device,
                            hidden_channels_dim=hidden_channels_dim,
                            pool_grid=pool_grid,
                            simple_proj=simple_proj,
                    ).to(device)
                    
                    w_path = scarica_pesi(f"moco_bot_weights_{moco_version}")               

                elif moco_ms:
                    model = MoCo2encoders_ms(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        hidden_channels_dim=hidden_channels_dim,
                        pool_grid=pool_grid,
                    ).to(device)

                    w_path = scarica_pesi(f"moco_2D_weights_{moco_version}") 
                
                else:
                    model = MoCo2encoders_2d(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        symmetric=False, device=device,
                        hidden_channels_dim=hidden_channels_dim,
                        simple_proj=simple_proj,
                        pool_grid=pool_grid,
                    ).to(device)

                    w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")

            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("OK -> modello MoCo caricato",  "\n", flush=True)

        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
            raise

        # embedding moco in tutti i punti selezionati, un solo passaggio
        # ritorna {punto: (q, k)}, q = ottico, k = sar
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            out = {punto: ([], []) for punto in punti}
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)

                    # encoder: latente autoencoder + pooling di moco
                    zq = model.encoder_q(im_q)
                    zk = model.encoder_k(im_k)
                    fq = {"encoder": model.pool_q(zq).flatten(1)}
                    fk = {"encoder": model.pool_k(zk).flatten(1)}

                    # pre_proj: conv + pooling (come contrastive_loss)
                    if model.simple_proj:
                        hq = model.pool_q(zq).flatten(1)
                        hk = model.pool_k(zk).flatten(1)
                    else:
                        hq = model.pool_q(model.conv_q(zq)).flatten(1)
                        hk = model.pool_k(model.conv_k(zk)).flatten(1)
                    fq["pre_proj"], fk["pre_proj"] = hq, hk

                    # proj_raw: proiettore senza normalizzazione
                    pq = model.proj_q(hq)
                    pk = model.proj_k(hk)
                    fq["proj_raw"], fk["proj_raw"] = pq, pk

                    # post_proj: proiettore + normalizzazione (uguale a contrastive_loss)
                    if moco_ms:
                        fq["post_proj"] = model.mean_shift(nn.functional.normalize(pq, dim=1), model.center_q)
                        fk["post_proj"] = model.mean_shift(nn.functional.normalize(pk, dim=1), model.center_k)
                    else:
                        fq["post_proj"] = nn.functional.normalize(pq, dim=1)
                        fk["post_proj"] = nn.functional.normalize(pk, dim=1)

                    for punto in punti:
                        out[punto][0].append(fq[punto].float().flatten(1).cpu())
                        out[punto][1].append(fk[punto].float().flatten(1).cpu())
            return {punto: (torch.cat(q).numpy(), torch.cat(k).numpy()) for punto, (q, k) in out.items()}

        def print_dim(nome, emb):
            dims = " | ".join(f"{punto} q {q.shape[1]} + k {k.shape[1]}" for punto, (q, k) in emb.items())
            print(f"OK -> {nome}: {len(next(iter(emb.values()))[0])} serie | {dims}", "\n", flush=True)

        # embedding serie di fit
        print("Embedding MoCo: serie di fit ocsvm", flush=True)
        emb = embed(normal_fit_ids)
        for punto in punti: E[punto]["fit"] = emb[punto]
        print_dim("normali fit", emb)

        # embedding serie thr
        print("Embedding MoCo: serie thr ocsvm", flush=True)
        emb = embed(normal_thr_ids)
        for punto in punti: E[punto]["thr"] = emb[punto]
        print_dim("normali thr", emb)

        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        emb = embed(normal_test_ids)
        for punto in punti: E[punto]["normal"] = emb[punto]
        print_dim("normali test", emb)

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
        emb = embed(anom_ids)
        for punto in punti: E[punto]["anom"] = emb[punto]
        print_dim("anomale", emb)

    # cartella radice, una sottocartella per punto
    out_root = f"/data/anomaly_results_ocsvm_{job_name}"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root)
    riepilogo = {}

    # anomaly detection ripetuta per ogni punto
    for punto in punti:
        print(f"===== {punto} =====", flush=True)
        q_fit, k_fit = E[punto]["fit"]
        q_thr, k_thr = E[punto]["thr"]
        q_normal, k_normal = E[punto]["normal"]
        q_anom, k_anom = E[punto]["anom"]

        # concatenazione
        w_fit = np.concatenate([q_fit, k_fit], axis=1).astype(np.float64)
        w_thr = np.concatenate([q_thr, k_thr], axis=1).astype(np.float64)
        w_normal = np.concatenate([q_normal, k_normal], axis=1).astype(np.float64)
        w_anom = np.concatenate([q_anom, k_anom],axis=1).astype(np.float64)

        if standardize:
            scaler = StandardScaler().fit(w_fit)
            w_fit, w_thr, w_normal, w_anom = scaler.transform(w_fit), scaler.transform(w_thr), scaler.transform(w_normal), scaler.transform(w_anom)

        if whitening == "lw":
            lw = LedoitWolf().fit(w_fit)
            lam, V = np.linalg.eigh(lw.covariance_)           
            W = V / np.sqrt(lam)                              
            z_fit = (w_fit - lw.location_) @ W
            z_thr = (w_thr - lw.location_) @ W
            z_normal = (w_normal - lw.location_) @ W
            z_anom = (w_anom - lw.location_) @ W
            n_componenti = w_fit.shape[1]
            shrinkage = float(lw.shrinkage_)
            print(f"Ledoit-Wolf: {n_componenti} dimensioni | shrinkage {shrinkage:.3f} | "
                  f"autovalore min {lam.min():.4f} max {lam.max():.2f}", flush=True)
        else:
            pw = PCA(n_components=pca_punto[punto], whiten=True).fit(w_fit)
            z_fit = pw.transform(w_fit)
            z_thr = pw.transform(w_thr)
            z_normal = pw.transform(w_normal)
            z_anom = pw.transform(w_anom)
            n_componenti = int(pw.n_components_)
            shrinkage = None
            print(f"PCA: {n_componenti} componenti su {w_fit.shape[1]} dimensioni", flush=True)

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
        out_dir = os.path.join(out_root, punto)
        os.makedirs(out_dir)

        # punteggi
        df = pd.DataFrame({
            "ID": normal_test_ids + anom_ids,
            "label": [0] * len(score_normal) + [1] * len(score_anom),
            "score": np.concatenate([score_normal, score_anom]),
        })
        df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
        pd.DataFrame({"ID": normal_thr_ids, "score": score_thr}).to_csv(os.path.join(out_dir, "scores_thr.csv"), index=False)


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
        print(flush=True)

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
            "feature_point": punto, "pca_components": pca_punto[punto], "n_componenti": n_componenti,
            "whitening": whitening, "shrinkage": shrinkage, "fit_on_train": fit_on_train,           
            "weights_version": weights_version or "ultima",            
            "ocsvm": {"nu": ocsvm_nu, "gamma": gamma, "dim": int(z_fit.shape[1]), "frac_outside_fit": frac_outside_fit},
            "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
            "auroc": float(auroc),
            "rules": rules,
        }
        with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        riepilogo[punto] = {"auroc": float(auroc), "n_componenti": n_componenti, "shrinkage": shrinkage}

    # riepilogo di tutti i punti (serve anche per old_vectors)
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump({"job_name": job_name, "moco_version": moco_version, "dataset_anomalies": dataset_anomalies,
                   "fit_on_train": fit_on_train, "whitening": whitening,
                   "weights_version": weights_version or "ultima",                            
                   "punti": riepilogo}, f, indent=2)

    print("===== RIEPILOGO =====", flush=True)
    for punto, r in riepilogo.items():
        print(f"{punto:9s}: auroc {r['auroc']:.4f} | {r['n_componenti']} componenti", flush=True)

    shutil.make_archive(out_root, "zip", out_root)
    project_work.log_artifact(name=f"anomaly_results_ocsvm_{job_name}", source=f"{out_root}.zip", kind="artifact")

    return "TERMINATO -> anomaly detection ocsvm"




@handler()
def ad_ocsvm4(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
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
    n_series_thr: int = 500,                        # usato solo con fit_on_train=False
    moco_attn: bool = False,

    do_encoder: bool = True,                        # latente autoencoder + pooling di moco
    do_pre_proj: bool = True,                       # dopo conv + pooling, prima del proiettore
    do_proj_raw: bool = True,                       # dopo il proiettore, prima della normalizzazione
    do_post_proj: bool = True,                      # dopo la normalizzazione (come contrastive_loss)

    pca_encoder: float | int | None = None,
    pca_pre_proj: float | int | None = None,
    pca_proj_raw: float | int | None = None,
    pca_post_proj: float | int | None = None,

    whitening: str = "pca",                         # "pca" | "ledoitwolf"
    fit_on_train: bool = True,                      # True: fit su tutte le serie di train di moco
    max_fit_series: int = 0,                        # 0 = tutte; altrimenti campione casuale (ocsvm lento con molte serie)

    dense: bool = False,                            # True -> modello di train_moco_dense
    do_dense: bool = False,                         # mappa densa (dense_q/dense_k) + pooling
    do_sim_stats: bool = False,                     # statistiche della mappa di similarità SAR-ottico
    pca_dense: float | int | None = None,
    pca_sim_stats: float | int | None = None,
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    # punti selezionati e componenti pca per punto
    punti = [p for p, do in (("encoder", do_encoder), ("pre_proj", do_pre_proj), ("proj_raw", do_proj_raw),
                             ("post_proj", do_post_proj), ("dense", do_dense), ("sim_stats", do_sim_stats)) if do]
    pca_punto = {"encoder": pca_encoder, "pre_proj": pca_pre_proj, "proj_raw": pca_proj_raw, "post_proj": pca_post_proj,
                 "dense": pca_dense, "sim_stats": pca_sim_stats}

    if (do_dense or do_sim_stats) and not dense:
        raise ValueError("do_dense e do_sim_stats richiedono dense=True")

    if not punti:
        raise ValueError("nessun punto selezionato: attiva almeno un do_*")
    
    if whitening not in ("pca", "lw"):
        raise ValueError(f"whitening non valido: {whitening}")
    print(f"punti: {', '.join(punti)} | whitening: {whitening} | fit_on_train: {fit_on_train}", "\n", flush=True)

    project_work = dh.get_project("floods")

    # embedding per punto -> E[punto] = {"fit": (q, k), "thr": (q, k), "normal": (q, k), "anom": (q, k)}
    E = {punto: {} for punto in punti}

    if old_vectors:
        print(f"Upload embedding: anomaly_results_ocsvm_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_ocsvm_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            old_summary = json.loads(z.read("summary.json"))
            mancanti = [punto for punto in punti if f"{punto}/embeddings.npz" not in z.namelist()]
            if mancanti:
                raise ValueError(f"punti non presenti in {old_vectors}: {mancanti}")
            for punto in punti:
                z.extract(f"{punto}/embeddings.npz", "/data/old_vectors")
                z.extract(f"{punto}/anomaly_results.json", "/data/old_vectors")

        moco_version = old_summary["moco_version"]
        dataset_anomalies = old_summary["dataset_anomalies"]
        fit_on_train = old_summary.get("fit_on_train", False)
        with open(f"/data/old_vectors/{punti[0]}/anomaly_results.json") as f:
            min_flood_area = json.load(f)["min_flood_area"]

        ids_rif = None
        for punto in punti:
            old = np.load(f"/data/old_vectors/{punto}/embeddings.npz")
            for g in ("fit", "thr", "normal", "anom"):
                E[punto][g] = (old[f"{g}_q"], old[f"{g}_k"])

            ids = {g: [str(ID) for ID in old[f"{g}_ids"]] for g in ("fit", "thr", "normal", "anom")}
            if ids_rif is None:
                ids_rif = ids
            elif ids != ids_rif:
                raise ValueError(f"il punto {punto} ha serie diverse dagli altri punti")
            print(f"OK -> {punto}: embedding caricati", flush=True)

        normal_fit_ids, normal_thr_ids = ids_rif["fit"], ids_rif["thr"]
        normal_test_ids, anom_ids = ids_rif["normal"], ids_rif["anom"]
        print(f"OK -> {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali, "
              f"{len(anom_ids)} anomale | moco {moco_version} | flood >= {min_flood_area:g}%", "\n", flush=True)

    else:
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

        normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]             # anomaly detection

        if fit_on_train:
            #fit su tutte le serie di train di moco
            if moco_attn:
                p_train = project_work.get_artifact(f"moco_attn_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
            else:
                p_train = project_work.get_artifact(f"moco_2D_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
            with open(p_train, 'r') as f:
                train_ids = json.load(f)

            normal_fit_ids = [ID for ID in train_ids if ID in normal_ids]                              # fit -> iperpiano
            if max_fit_series > 0 and len(normal_fit_ids) > max_fit_series:
                random.seed(1)
                normal_fit_ids = random.sample(normal_fit_ids, max_fit_series)

            test_set = set(normal_test_ids)
            normal_thr_ids = [ID for ID in shuffled_test_ids if ID not in test_set]                    # calcolo soglie
        else:
            normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]                        # fit -> iperpiano
            normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]          # calcolo soglie

        assert not (set(normal_fit_ids) & set(normal_thr_ids)), "serie in comune tra fit e soglia"
        assert not (set(normal_fit_ids) & set(normal_test_ids)), "serie in comune tra fit e test"
        assert not (set(normal_thr_ids) & set(normal_test_ids)), "serie in comune tra soglia e test"

        print(f"OK -> {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali", "\n", flush=True)

        print("calcolo vettori MoCo", flush = True)

        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"

        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids)}
        print(f"Estrazione {len(wanted_normal_series)} serie normali", flush=True)
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
                ).to(device)

                w_path = project_work.get_artifact(f"moco_attn_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)

            else:
                encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
                encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)

                if dense:
                    model = MoCo2encoders_dense(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        hidden_channels_dim=hidden_channels_dim,
                        pool_grid=pool_grid,
                    ).to(device)
                else:
                    model = MoCo2encoders_2d(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        symmetric=False, device=device,
                        hidden_channels_dim=hidden_channels_dim,
                        simple_proj=simple_proj,
                        pool_grid=pool_grid,
                    ).to(device)
                print(f"classe: {type(model).__name__}", flush=True)

                w_path = project_work.get_artifact(f"moco_2D_weights_{moco_version}").download("/data/moco_weights.pth", overwrite=True)

            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("OK -> modello MoCo caricato",  "\n", flush=True)

        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
            raise

        # embedding moco in tutti i punti selezionati, un solo passaggio
        # ritorna {punto: (q, k)}, q = ottico, k = sar
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            out = {punto: ([], []) for punto in punti}
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)

                    # encoder: latente autoencoder + pooling di moco
                    zq = model.encoder_q(im_q)
                    zk = model.encoder_k(im_k)
                    fq = {"encoder": model.pool_q(zq).flatten(1)}
                    fk = {"encoder": model.pool_k(zk).flatten(1)}

                    # pre_proj: conv + pooling (come contrastive_loss)
                    if getattr(model, "simple_proj", False):
                        hq = model.pool_q(zq).flatten(1)
                        hk = model.pool_k(zk).flatten(1)
                    else:
                        cq = model.conv_q(zq)                                    # (B, C, 64, 64)
                        ck = model.conv_k(zk)
                        hq = model.pool_q(cq).flatten(1)
                        hk = model.pool_k(ck).flatten(1)
                    fq["pre_proj"], fk["pre_proj"] = hq, hk

                    # proj_raw: proiettore senza normalizzazione
                    pq = model.proj_q(hq)
                    pk = model.proj_k(hk)
                    fq["proj_raw"], fk["proj_raw"] = pq, pk

                    # post_proj: proiettore + normalizzazione (uguale a contrastive_loss)
                    fq["post_proj"] = nn.functional.normalize(pq, dim=1)
                    fk["post_proj"] = nn.functional.normalize(pk, dim=1)

                    # punti densi: solo per il modello di train_moco_dense
                    if dense:
                        dq = nn.functional.normalize(model.dense_q(cq), dim=1)        # (B, dim, 64, 64)
                        dk = nn.functional.normalize(model.dense_k(ck), dim=1)

                        # dense: mappa densa ridotta con il pooling, come pre_proj
                        fq["dense"] = model.pool_q(dq).flatten(1)
                        fk["dense"] = model.pool_k(dk).flatten(1)

                        # sim_stats: statistiche della similarità SAR-ottico posizione per posizione
                        s = (dq * dk).sum(dim=1).flatten(1).float()                     # (B, 4096)
                        fq["sim_stats"] = torch.stack([
                            s.min(dim=1).values,
                            torch.quantile(s, 0.01, dim=1),
                            torch.quantile(s, 0.05, dim=1),
                            torch.quantile(s, 0.10, dim=1),
                            s.mean(dim=1),
                            s.std(dim=1),
                        ], dim=1)                                                        # (B, 6)
                        fk["sim_stats"] = s[:, :0]

                    for punto in punti:
                        out[punto][0].append(fq[punto].float().flatten(1).cpu())
                        out[punto][1].append(fk[punto].float().flatten(1).cpu())
            return {punto: (torch.cat(q).numpy(), torch.cat(k).numpy()) for punto, (q, k) in out.items()}

        def print_dim(nome, emb):
            dims = " | ".join(f"{punto} q {q.shape[1]} + k {k.shape[1]}" for punto, (q, k) in emb.items())
            print(f"OK -> {nome}: {len(next(iter(emb.values()))[0])} serie | {dims}", "\n", flush=True)

        # embedding serie di fit
        print("Embedding MoCo: serie di fit ocsvm", flush=True)
        emb = embed(normal_fit_ids)
        for punto in punti: E[punto]["fit"] = emb[punto]
        print_dim("normali fit", emb)

        # embedding serie thr
        print("Embedding MoCo: serie thr ocsvm", flush=True)
        emb = embed(normal_thr_ids)
        for punto in punti: E[punto]["thr"] = emb[punto]
        print_dim("normali thr", emb)

        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        emb = embed(normal_test_ids)
        for punto in punti: E[punto]["normal"] = emb[punto]
        print_dim("normali test", emb)

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
        emb = embed(anom_ids)
        for punto in punti: E[punto]["anom"] = emb[punto]
        print_dim("anomale", emb)

    # cartella radice, una sottocartella per punto
    out_root = f"/data/anomaly_results_ocsvm_{job_name}"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root)
    riepilogo = {}

    # anomaly detection ripetuta per ogni punto
    for punto in punti:
        print(f"===== {punto} =====", flush=True)
        q_fit, k_fit = E[punto]["fit"]
        q_thr, k_thr = E[punto]["thr"]
        q_normal, k_normal = E[punto]["normal"]
        q_anom, k_anom = E[punto]["anom"]

        # concatenazione
        w_fit = np.concatenate([q_fit, k_fit], axis=1).astype(np.float64)
        w_thr = np.concatenate([q_thr, k_thr], axis=1).astype(np.float64)
        w_normal = np.concatenate([q_normal, k_normal], axis=1).astype(np.float64)
        w_anom = np.concatenate([q_anom, k_anom],axis=1).astype(np.float64)

        if standardize:
            scaler = StandardScaler().fit(w_fit)
            w_fit, w_thr, w_normal, w_anom = scaler.transform(w_fit), scaler.transform(w_thr), scaler.transform(w_normal), scaler.transform(w_anom)

        if whitening == "lw":
            lw = LedoitWolf().fit(w_fit)
            lam, V = np.linalg.eigh(lw.covariance_)           
            W = V / np.sqrt(lam)                              
            z_fit = (w_fit - lw.location_) @ W
            z_thr = (w_thr - lw.location_) @ W
            z_normal = (w_normal - lw.location_) @ W
            z_anom = (w_anom - lw.location_) @ W
            n_componenti = w_fit.shape[1]
            shrinkage = float(lw.shrinkage_)
            print(f"Ledoit-Wolf: {n_componenti} dimensioni | shrinkage {shrinkage:.3f} | "
                  f"autovalore min {lam.min():.4f} max {lam.max():.2f}", flush=True)
        else:
            pw = PCA(n_components=pca_punto[punto], whiten=True).fit(w_fit)
            z_fit = pw.transform(w_fit)
            z_thr = pw.transform(w_thr)
            z_normal = pw.transform(w_normal)
            z_anom = pw.transform(w_anom)
            n_componenti = int(pw.n_components_)
            shrinkage = None
            print(f"PCA: {n_componenti} componenti su {w_fit.shape[1]} dimensioni", flush=True)

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
        out_dir = os.path.join(out_root, punto)
        os.makedirs(out_dir)

        # punteggi
        df = pd.DataFrame({
            "ID": normal_test_ids + anom_ids,
            "label": [0] * len(score_normal) + [1] * len(score_anom),
            "score": np.concatenate([score_normal, score_anom]),
        })
        df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
        pd.DataFrame({"ID": normal_thr_ids, "score": score_thr}).to_csv(os.path.join(out_dir, "scores_thr.csv"), index=False)


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
        print(flush=True)

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
            "feature_point": punto, "pca_components": pca_punto[punto], "n_componenti": n_componenti,
            "whitening": whitening, "shrinkage": shrinkage, "fit_on_train": fit_on_train,                   
            "ocsvm": {"nu": ocsvm_nu, "gamma": gamma, "dim": int(z_fit.shape[1]), "frac_outside_fit": frac_outside_fit},
            "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
            "auroc": float(auroc),
            "rules": rules,
        }
        with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        riepilogo[punto] = {"auroc": float(auroc), "n_componenti": n_componenti, "shrinkage": shrinkage}

    # riepilogo di tutti i punti (serve anche per old_vectors)
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump({"job_name": job_name, "moco_version": moco_version, "dataset_anomalies": dataset_anomalies,
                   "fit_on_train": fit_on_train, "whitening": whitening,                                     
                   "punti": riepilogo}, f, indent=2)

    print("===== RIEPILOGO =====", flush=True)
    for punto, r in riepilogo.items():
        print(f"{punto:9s}: auroc {r['auroc']:.4f} | {r['n_componenti']} componenti", flush=True)

    shutil.make_archive(out_root, "zip", out_root)
    project_work.log_artifact(name=f"anomaly_results_ocsvm_{job_name}", source=f"{out_root}.zip", kind="artifact")

    return "TERMINATO -> anomaly detection ocsvm"





@handler()
def ad_ocsvm5(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
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
    n_series_thr: int = 500,                        # usato solo con fit_on_train=False
    moco_attn: bool = False,
    moco_bot: bool = False,
    moco_ms: bool = False,
    moco_ms_pre: bool = False,                      # NUOVO: modello MoCo2encoders_ms_pre

    do_encoder: bool = True,                        # latente autoencoder + pooling di moco
    do_pre_proj: bool = True,                       # dopo conv + pooling, prima del proiettore
    do_pre_proj_ms: bool = True,                   # NUOVO: pre_proj normalizzato e spostato rispetto al centro (solo ms_pre)
    do_proj_raw: bool = True,                       # dopo il proiettore, prima della normalizzazione
    do_post_proj: bool = True,                      # dopo la normalizzazione (come contrastive_loss)

    pca_encoder: float | int | None = None,
    pca_pre_proj: float | int | None = None,
    pca_pre_proj_ms: float | int | None = None,     # NUOVO
    pca_proj_raw: float | int | None = None,
    pca_post_proj: float | int | None = None,

    whitening: str = "pca",                         # "pca" | "lw"
    fit_on_train: bool = True,                      # True: fit su tutte le serie di train di moco
    max_fit_series: int = 0,                        # 0 = tutte; altrimenti campione casuale (ocsvm lento con molte serie)

    weights_version: str = "",
):
    from sklearn.covariance import LedoitWolf

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    # punti selezionati e componenti pca per punto
    punti = [p for p, do in (("encoder", do_encoder), ("pre_proj", do_pre_proj), ("pre_proj_ms", do_pre_proj_ms),   # MODIFICATO
                             ("proj_raw", do_proj_raw), ("post_proj", do_post_proj)) if do]
    pca_punto = {"encoder": pca_encoder, "pre_proj": pca_pre_proj, "pre_proj_ms": pca_pre_proj_ms,                  # MODIFICATO
                 "proj_raw": pca_proj_raw, "post_proj": pca_post_proj}

    if not punti:
        raise ValueError("nessun punto selezionato: attiva almeno un do_*")
    if whitening not in ("pca", "lw"):
        raise ValueError(f"whitening non valido: {whitening}")
    if do_pre_proj_ms and not moco_ms_pre:                                                                          # NUOVO
        raise ValueError("do_pre_proj_ms richiede moco_ms_pre=True")
    if sum([moco_attn, moco_bot, moco_ms, moco_ms_pre]) > 1:                                                        # NUOVO
        raise ValueError("indica al massimo uno tra moco_attn, moco_bot, moco_ms, moco_ms_pre")
    print(f"punti: {', '.join(punti)} | whitening: {whitening} | fit_on_train: {fit_on_train}", "\n", flush=True)

    project_work = dh.get_project("floods")

    # embedding per punto -> E[punto] = {"fit": (q, k), "thr": (q, k), "normal": (q, k), "anom": (q, k)}
    E = {punto: {} for punto in punti}

    if old_vectors:
        print(f"Upload embedding: anomaly_results_ocsvm_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_ocsvm_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            old_summary = json.loads(z.read("summary.json"))
            mancanti = [punto for punto in punti if f"{punto}/embeddings.npz" not in z.namelist()]
            if mancanti:
                raise ValueError(f"punti non presenti in {old_vectors}: {mancanti}")
            for punto in punti:
                z.extract(f"{punto}/embeddings.npz", "/data/old_vectors")
                z.extract(f"{punto}/anomaly_results.json", "/data/old_vectors")

        moco_version = old_summary["moco_version"]
        dataset_anomalies = old_summary["dataset_anomalies"]
        fit_on_train = old_summary.get("fit_on_train", False)
        with open(f"/data/old_vectors/{punti[0]}/anomaly_results.json") as f:
            min_flood_area = json.load(f)["min_flood_area"]

        ids_rif = None
        for punto in punti:
            old = np.load(f"/data/old_vectors/{punto}/embeddings.npz")
            for g in ("fit", "thr", "normal", "anom"):
                E[punto][g] = (old[f"{g}_q"], old[f"{g}_k"])

            ids = {g: [str(ID) for ID in old[f"{g}_ids"]] for g in ("fit", "thr", "normal", "anom")}
            if ids_rif is None:
                ids_rif = ids
            elif ids != ids_rif:
                raise ValueError(f"il punto {punto} ha serie diverse dagli altri punti")
            print(f"OK -> {punto}: embedding caricati", flush=True)

        normal_fit_ids, normal_thr_ids = ids_rif["fit"], ids_rif["thr"]
        normal_test_ids, anom_ids = ids_rif["normal"], ids_rif["anom"]
        print(f"OK -> {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali, "
              f"{len(anom_ids)} anomale | moco {moco_version} | flood >= {min_flood_area:g}%", "\n", flush=True)

    else:
        project_data = dh.get_project("datasets")

        # SPOSTATO: caricamento pesi moco prima dei dataset (veloce: se fallisce, il job si ferma subito)
        try:
            def scarica_pesi(nome_artifact):
                if weights_version:
                    art = dh.get_artifact(nome_artifact, project="floods", entity_id=weights_version)
                else:
                    art = project_work.get_artifact(nome_artifact)
                print(f"pesi: {nome_artifact} (versione {weights_version or 'ultima'})", flush=True)
                return art.download("/data/moco_weights.pth", overwrite=True)

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
                ).to(device)
                w_path = scarica_pesi(f"moco_attn_weights_{moco_version}")

            else:
                encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
                encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)

                if moco_bot:
                    model = MoCo2encoders_Bot(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        symmetric=False, device=device,
                        hidden_channels_dim=hidden_channels_dim,
                        pool_grid=pool_grid,
                        simple_proj=simple_proj,
                    ).to(device)
                    w_path = scarica_pesi(f"moco_bot_weights_{moco_version}")

                elif moco_ms:
                    model = MoCo2encoders_ms(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        hidden_channels_dim=hidden_channels_dim,
                        pool_grid=pool_grid,
                    ).to(device)
                    w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")

                elif moco_ms_pre:                                                                    # NUOVO
                    model = MoCo2encoders_ms_pre(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        hidden_channels_dim=hidden_channels_dim,
                        pool_grid=pool_grid,
                    ).to(device)
                    w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")

                else:
                    model = MoCo2encoders_2d(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        symmetric=False, device=device,
                        hidden_channels_dim=hidden_channels_dim,
                        simple_proj=simple_proj,
                        pool_grid=pool_grid,
                    ).to(device)
                    w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")

            print(f"classe: {type(model).__name__}", flush=True)                                     # NUOVO

            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

            # NUOVO: caricamento strict, ma con l'elenco chiaro delle chiavi che non tornano (code e centri compresi)
            mancanti, inattesi = model.load_state_dict(state_dict, strict=False)
            if mancanti or inattesi:
                print(f"chiavi mancanti nel checkpoint: {mancanti}", flush=True)
                print(f"chiavi inattese nel checkpoint: {inattesi}", flush=True)
                raise ValueError(f"il checkpoint di {moco_version} non corrisponde alla classe {type(model).__name__}")

            if moco_ms or moco_ms_pre:                                                               # NUOVO: controllo dei centri
                print(f"centri: |c_q| {model.center_q.norm():.3f} | |c_k| {model.center_k.norm():.3f}", flush=True)
                if model.center_q.norm() == 0 or model.center_k.norm() == 0:
                    raise ValueError("centri a zero nel checkpoint: il modello non è stato allenato con la mean-shift")

            model.eval()
            print("OK -> modello MoCo caricato", "\n", flush=True)

        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
            raise

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

        # liste ids dal training di moco
        prefisso = "moco_attn" if moco_attn else ("moco_bot" if moco_bot else "moco_2D")             # MODIFICATO: un solo prefisso
        p = project_work.get_artifact(f"{prefisso}_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)

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

        normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]             # anomaly detection

        if fit_on_train:
            p_train = project_work.get_artifact(f"{prefisso}_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
            with open(p_train, 'r') as f:
                train_ids = json.load(f)

            normal_fit_ids = [ID for ID in train_ids if ID in normal_ids]                              # fit -> iperpiano
            if max_fit_series > 0 and len(normal_fit_ids) > max_fit_series:
                random.seed(1)
                normal_fit_ids = random.sample(normal_fit_ids, max_fit_series)

            test_set = set(normal_test_ids)
            normal_thr_ids = [ID for ID in shuffled_test_ids if ID not in test_set]                    # calcolo soglie
        else:
            normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]                        # fit -> iperpiano
            normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]          # calcolo soglie

        assert not (set(normal_fit_ids) & set(normal_thr_ids)), "serie in comune tra fit e soglia"
        assert not (set(normal_fit_ids) & set(normal_test_ids)), "serie in comune tra fit e test"
        assert not (set(normal_thr_ids) & set(normal_test_ids)), "serie in comune tra soglia e test"

        print(f"OK -> {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali", "\n", flush=True)

        print("calcolo vettori MoCo", flush = True)

        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"

        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids)}
        print(f"Estrazione {len(wanted_normal_series)} serie normali", flush=True)
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)

        # embedding moco in tutti i punti selezionati, un solo passaggio
        # ritorna {punto: (q, k)}, q = ottico, k = sar
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            out = {punto: ([], []) for punto in punti}
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)

                    # encoder: latente autoencoder + pooling di moco
                    zq = model.encoder_q(im_q)
                    zk = model.encoder_k(im_k)
                    fq = {"encoder": model.pool_q(zq).flatten(1)}
                    fk = {"encoder": model.pool_k(zk).flatten(1)}

                    # pre_proj: conv + pooling (come contrastive_loss)
                    if model.simple_proj:
                        hq = model.pool_q(zq).flatten(1)
                        hk = model.pool_k(zk).flatten(1)
                    else:
                        hq = model.pool_q(model.conv_q(zq)).flatten(1)
                        hk = model.pool_k(model.conv_k(zk)).flatten(1)
                    fq["pre_proj"], fk["pre_proj"] = hq, hk

                    # NUOVO: pre_proj normalizzato e spostato rispetto al centro (spazio della loss di ms_pre)
                    if moco_ms_pre:
                        fq["pre_proj_ms"] = model.mean_shift(nn.functional.normalize(hq.float(), dim=1), model.center_q)
                        fk["pre_proj_ms"] = model.mean_shift(nn.functional.normalize(hk.float(), dim=1), model.center_k)

                    # proj_raw: proiettore senza normalizzazione
                    pq = model.proj_q(hq)
                    pk = model.proj_k(hk)
                    fq["proj_raw"], fk["proj_raw"] = pq, pk

                    # post_proj: proiettore + normalizzazione (+ spostamento per il modello ms)
                    if moco_ms:
                        fq["post_proj"] = model.mean_shift(nn.functional.normalize(pq, dim=1), model.center_q)
                        fk["post_proj"] = model.mean_shift(nn.functional.normalize(pk, dim=1), model.center_k)
                    else:
                        fq["post_proj"] = nn.functional.normalize(pq, dim=1)
                        fk["post_proj"] = nn.functional.normalize(pk, dim=1)

                    for punto in punti:
                        out[punto][0].append(fq[punto].float().flatten(1).cpu())
                        out[punto][1].append(fk[punto].float().flatten(1).cpu())
            return {punto: (torch.cat(q).numpy(), torch.cat(k).numpy()) for punto, (q, k) in out.items()}

        def print_dim(nome, emb):
            dims = " | ".join(f"{punto} q {q.shape[1]} + k {k.shape[1]}" for punto, (q, k) in emb.items())
            print(f"OK -> {nome}: {len(next(iter(emb.values()))[0])} serie | {dims}", "\n", flush=True)

        # embedding serie di fit
        print("Embedding MoCo: serie di fit ocsvm", flush=True)
        emb = embed(normal_fit_ids)
        for punto in punti: E[punto]["fit"] = emb[punto]
        print_dim("normali fit", emb)

        # embedding serie thr
        print("Embedding MoCo: serie thr ocsvm", flush=True)
        emb = embed(normal_thr_ids)
        for punto in punti: E[punto]["thr"] = emb[punto]
        print_dim("normali thr", emb)

        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        emb = embed(normal_test_ids)
        for punto in punti: E[punto]["normal"] = emb[punto]
        print_dim("normali test", emb)

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
        emb = embed(anom_ids)
        for punto in punti: E[punto]["anom"] = emb[punto]
        print_dim("anomale", emb)

    # cartella radice, una sottocartella per punto
    out_root = f"/data/anomaly_results_ocsvm_{job_name}"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root)
    riepilogo = {}

    # anomaly detection ripetuta per ogni punto
    for punto in punti:
        print(f"===== {punto} =====", flush=True)
        q_fit, k_fit = E[punto]["fit"]
        q_thr, k_thr = E[punto]["thr"]
        q_normal, k_normal = E[punto]["normal"]
        q_anom, k_anom = E[punto]["anom"]

        # concatenazione
        w_fit = np.concatenate([q_fit, k_fit], axis=1).astype(np.float64)
        w_thr = np.concatenate([q_thr, k_thr], axis=1).astype(np.float64)
        w_normal = np.concatenate([q_normal, k_normal], axis=1).astype(np.float64)
        w_anom = np.concatenate([q_anom, k_anom],axis=1).astype(np.float64)

        if standardize:
            scaler = StandardScaler().fit(w_fit)
            w_fit, w_thr, w_normal, w_anom = scaler.transform(w_fit), scaler.transform(w_thr), scaler.transform(w_normal), scaler.transform(w_anom)

        if whitening == "lw":
            lw = LedoitWolf().fit(w_fit)
            lam, V = np.linalg.eigh(lw.covariance_)
            W = V / np.sqrt(lam)
            z_fit = (w_fit - lw.location_) @ W
            z_thr = (w_thr - lw.location_) @ W
            z_normal = (w_normal - lw.location_) @ W
            z_anom = (w_anom - lw.location_) @ W
            n_componenti = w_fit.shape[1]
            shrinkage = float(lw.shrinkage_)
            print(f"Ledoit-Wolf: {n_componenti} dimensioni | shrinkage {shrinkage:.3f} | "
                  f"autovalore min {lam.min():.4f} max {lam.max():.2f}", flush=True)
        else:
            pw = PCA(n_components=pca_punto[punto], whiten=True).fit(w_fit)
            z_fit = pw.transform(w_fit)
            z_thr = pw.transform(w_thr)
            z_normal = pw.transform(w_normal)
            z_anom = pw.transform(w_anom)
            n_componenti = int(pw.n_components_)
            shrinkage = None
            print(f"PCA: {n_componenti} componenti su {w_fit.shape[1]} dimensioni", flush=True)

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
        out_dir = os.path.join(out_root, punto)
        os.makedirs(out_dir)

        # punteggi
        df = pd.DataFrame({
            "ID": normal_test_ids + anom_ids,
            "label": [0] * len(score_normal) + [1] * len(score_anom),
            "score": np.concatenate([score_normal, score_anom]),
        })
        df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
        pd.DataFrame({"ID": normal_thr_ids, "score": score_thr}).to_csv(os.path.join(out_dir, "scores_thr.csv"), index=False)

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
        print(flush=True)

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
            "feature_point": punto, "pca_components": pca_punto[punto], "n_componenti": n_componenti,
            "whitening": whitening, "shrinkage": shrinkage, "fit_on_train": fit_on_train,
            "weights_version": weights_version or "ultima",
            "ocsvm": {"nu": ocsvm_nu, "gamma": gamma, "dim": int(z_fit.shape[1]), "frac_outside_fit": frac_outside_fit},
            "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
            "auroc": float(auroc),
            "rules": rules,
        }
        with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        riepilogo[punto] = {"auroc": float(auroc), "n_componenti": n_componenti, "shrinkage": shrinkage}

    # riepilogo di tutti i punti (serve anche per old_vectors)
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump({"job_name": job_name, "moco_version": moco_version, "dataset_anomalies": dataset_anomalies,
                   "fit_on_train": fit_on_train, "whitening": whitening,
                   "weights_version": weights_version or "ultima",
                   "punti": riepilogo}, f, indent=2)

    print("===== RIEPILOGO =====", flush=True)
    for punto, r in riepilogo.items():
        print(f"{punto:11s}: auroc {r['auroc']:.4f} | {r['n_componenti']} componenti", flush=True)

    shutil.make_archive(out_root, "zip", out_root)
    project_work.log_artifact(name=f"anomaly_results_ocsvm_{job_name}", source=f"{out_root}.zip", kind="artifact")

    return "TERMINATO -> anomaly detection ocsvm"



@handler()
def ad_ocsvm6(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
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
    n_series_thr: int = 500,                        # usato solo con fit_on_train=False
    moco_attn: bool = False,
    moco_bot: bool = False,
    moco_ms: bool = False,
    moco_ms_pre: bool = False,
    moco_dense: bool = False,                       # NUOVO: modello MoCo2encoders_dense

    do_encoder: bool = True,                        # latente autoencoder + pooling di moco
    do_encoder_fine: bool = False,                  # NUOVO: latente autoencoder + pooling pool_fine
    do_pre_proj: bool = True,                       # dopo conv + pooling, prima del proiettore
    do_pre_proj_fine: bool = False,                 # NUOVO: dopo conv + pooling pool_fine
    do_pre_proj_ms: bool = False,                   # pre_proj normalizzato e spostato rispetto al centro (solo ms_pre)
    do_dense: bool = False,                         # NUOVO: ramo denso normalizzato per cella + pooling di moco (solo dense)
    do_proj_raw: bool = True,                       # dopo il proiettore, prima della normalizzazione
    do_post_proj: bool = True,                      # dopo la normalizzazione (come contrastive_loss)

    pca_encoder: float | int | None = None,
    pca_encoder_fine: float | int | None = None,
    pca_pre_proj: float | int | None = None,
    pca_pre_proj_fine: float | int | None = 0.99,   # 16384 dim con pool_fine=8: per l'OC-SVM conviene ridurre
    pca_pre_proj_ms: float | int | None = None,
    pca_dense: float | int | None = None,
    pca_proj_raw: float | int | None = None,
    pca_post_proj: float | int | None = None,

    pool_fine: int = 8,                             # NUOVO: griglia del pooling dei punti *_fine (solo estrazione)

    whitening: str = "pca",                         # "pca" | "lw"
    fit_on_train: bool = True,                      # True: fit su tutte le serie di train di moco
    max_fit_series: int = 0,                        # 0 = tutte; altrimenti campione casuale (ocsvm lento con molte serie)

    weights_version: str = "",
):
    from sklearn.covariance import LedoitWolf

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    # punti selezionati e componenti pca per punto
    punti = [p for p, do in (("encoder", do_encoder), ("encoder_fine", do_encoder_fine),
                             ("pre_proj", do_pre_proj), ("pre_proj_fine", do_pre_proj_fine),
                             ("pre_proj_ms", do_pre_proj_ms), ("dense", do_dense),
                             ("proj_raw", do_proj_raw), ("post_proj", do_post_proj)) if do]
    pca_punto = {"encoder": pca_encoder, "encoder_fine": pca_encoder_fine,
                 "pre_proj": pca_pre_proj, "pre_proj_fine": pca_pre_proj_fine,
                 "pre_proj_ms": pca_pre_proj_ms, "dense": pca_dense,
                 "proj_raw": pca_proj_raw, "post_proj": pca_post_proj}

    if not punti:
        raise ValueError("nessun punto selezionato: attiva almeno un do_*")
    if whitening not in ("pca", "lw"):
        raise ValueError(f"whitening non valido: {whitening}")
    if do_pre_proj_ms and not moco_ms_pre:
        raise ValueError("do_pre_proj_ms richiede moco_ms_pre=True")
    if do_dense and not moco_dense:
        raise ValueError("do_dense richiede moco_dense=True")
    if sum([moco_attn, moco_bot, moco_ms, moco_ms_pre, moco_dense]) > 1:
        raise ValueError("indica al massimo uno tra moco_attn, moco_bot, moco_ms, moco_ms_pre, moco_dense")
    if (do_encoder_fine or do_pre_proj_fine) and pool_fine < 1:
        raise ValueError("pool_fine deve essere >= 1")
    print(f"punti: {', '.join(punti)} | whitening: {whitening} | fit_on_train: {fit_on_train} | "
          f"pool_grid: {pool_grid} | pool_fine: {pool_fine}", "\n", flush=True)

    project_work = dh.get_project("floods")

    # embedding per punto -> E[punto] = {"fit": (q, k), "thr": (q, k), "normal": (q, k), "anom": (q, k)}
    E = {punto: {} for punto in punti}

    if old_vectors:
        print(f"Upload embedding: anomaly_results_ocsvm_{old_vectors}", flush=True)
        p = project_work.get_artifact(f"anomaly_results_ocsvm_{old_vectors}").download("/data/old_vectors.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            old_summary = json.loads(z.read("summary.json"))
            mancanti = [punto for punto in punti if f"{punto}/embeddings.npz" not in z.namelist()]
            if mancanti:
                raise ValueError(f"punti non presenti in {old_vectors}: {mancanti}")
            for punto in punti:
                z.extract(f"{punto}/embeddings.npz", "/data/old_vectors")
                z.extract(f"{punto}/anomaly_results.json", "/data/old_vectors")

        moco_version = old_summary["moco_version"]
        dataset_anomalies = old_summary["dataset_anomalies"]
        fit_on_train = old_summary.get("fit_on_train", False)
        pool_fine = old_summary.get("pool_fine", pool_fine)
        with open(f"/data/old_vectors/{punti[0]}/anomaly_results.json") as f:
            min_flood_area = json.load(f)["min_flood_area"]

        ids_rif = None
        for punto in punti:
            old = np.load(f"/data/old_vectors/{punto}/embeddings.npz")
            for g in ("fit", "thr", "normal", "anom"):
                E[punto][g] = (old[f"{g}_q"], old[f"{g}_k"])

            ids = {g: [str(ID) for ID in old[f"{g}_ids"]] for g in ("fit", "thr", "normal", "anom")}
            if ids_rif is None:
                ids_rif = ids
            elif ids != ids_rif:
                raise ValueError(f"il punto {punto} ha serie diverse dagli altri punti")
            print(f"OK -> {punto}: embedding caricati", flush=True)

        normal_fit_ids, normal_thr_ids = ids_rif["fit"], ids_rif["thr"]
        normal_test_ids, anom_ids = ids_rif["normal"], ids_rif["anom"]
        print(f"OK -> {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali, "
              f"{len(anom_ids)} anomale | moco {moco_version} | flood >= {min_flood_area:g}%", "\n", flush=True)

    else:
        project_data = dh.get_project("datasets")

        # caricamento pesi moco prima dei dataset (veloce: se fallisce, il job si ferma subito)
        try:
            def scarica_pesi(nome_artifact):
                if weights_version:
                    art = dh.get_artifact(nome_artifact, project="floods", entity_id=weights_version)
                else:
                    art = project_work.get_artifact(nome_artifact)
                print(f"pesi: {nome_artifact} (versione {weights_version or 'ultima'})", flush=True)
                return art.download("/data/moco_weights.pth", overwrite=True)

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
                ).to(device)
                w_path = scarica_pesi(f"moco_attn_weights_{moco_version}")

            else:
                encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
                encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)

                if moco_bot:
                    model = MoCo2encoders_Bot(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        symmetric=False, device=device,
                        hidden_channels_dim=hidden_channels_dim,
                        pool_grid=pool_grid,
                        simple_proj=simple_proj,
                    ).to(device)
                    w_path = scarica_pesi(f"moco_bot_weights_{moco_version}")

                elif moco_ms:
                    model = MoCo2encoders_ms(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        hidden_channels_dim=hidden_channels_dim,
                        pool_grid=pool_grid,
                    ).to(device)
                    w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")

                elif moco_ms_pre:
                    model = MoCo2encoders_ms_pre(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        hidden_channels_dim=hidden_channels_dim,
                        pool_grid=pool_grid,
                    ).to(device)
                    w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")

                elif moco_dense:
                    # usa gli stessi argomenti del costruttore con cui è stata allenata quella versione
                    model = MoCo2encoders_dense(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        hidden_channels_dim=hidden_channels_dim,
                        pool_grid=pool_grid,
                    ).to(device)
                    w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")

                else:
                    model = MoCo2encoders_2d(
                        base_encoder_q=encoderOPT.encoder,
                        base_encoder_k=encoderSAR.encoder,
                        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                        symmetric=False, device=device,
                        hidden_channels_dim=hidden_channels_dim,
                        simple_proj=simple_proj,
                        pool_grid=pool_grid,
                    ).to(device)
                    w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")

            print(f"classe: {type(model).__name__}", flush=True)

            checkpoint = torch.load(w_path, map_location=device)
            state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

            # caricamento strict, con l'elenco delle chiavi che non tornano (code e centri compresi)
            mancanti, inattesi = model.load_state_dict(state_dict, strict=False)
            if mancanti or inattesi:
                print(f"chiavi mancanti nel checkpoint: {mancanti}", flush=True)
                print(f"chiavi inattese nel checkpoint: {inattesi}", flush=True)
                raise ValueError(f"il checkpoint di {moco_version} non corrisponde alla classe {type(model).__name__}")

            if moco_ms or moco_ms_pre:
                print(f"centri: |c_q| {model.center_q.norm():.3f} | |c_k| {model.center_k.norm():.3f}", flush=True)
                if model.center_q.norm() == 0 or model.center_k.norm() == 0:
                    raise ValueError("centri a zero nel checkpoint: il modello non è stato allenato con la mean-shift")

            model.eval()
            usa_conv = not getattr(model, "simple_proj", False)        # MoCo2encoders_dense non ha simple_proj
            print("OK -> modello MoCo caricato", "\n", flush=True)

        except Exception as e:
            print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
            raise

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

        # liste ids dal training di moco
        prefisso = "moco_attn" if moco_attn else ("moco_bot" if moco_bot else "moco_2D")
        p = project_work.get_artifact(f"{prefisso}_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)

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

        normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]             # anomaly detection

        if fit_on_train:
            p_train = project_work.get_artifact(f"{prefisso}_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
            with open(p_train, 'r') as f:
                train_ids = json.load(f)

            normal_fit_ids = [ID for ID in train_ids if ID in normal_ids]                              # fit -> iperpiano
            if max_fit_series > 0 and len(normal_fit_ids) > max_fit_series:
                random.seed(1)
                normal_fit_ids = random.sample(normal_fit_ids, max_fit_series)

            test_set = set(normal_test_ids)
            normal_thr_ids = [ID for ID in shuffled_test_ids if ID not in test_set]                    # calcolo soglie
        else:
            normal_fit_ids  = shuffled_test_ids[:num_fit_series - n_series_thr]                        # fit -> iperpiano
            normal_thr_ids  = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]          # calcolo soglie

        assert not (set(normal_fit_ids) & set(normal_thr_ids)), "serie in comune tra fit e soglia"
        assert not (set(normal_fit_ids) & set(normal_test_ids)), "serie in comune tra fit e test"
        assert not (set(normal_thr_ids) & set(normal_test_ids)), "serie in comune tra soglia e test"

        print(f"OK -> {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali", "\n", flush=True)

        print("calcolo vettori MoCo", flush = True)

        # estrazione serie normal_ids, fit_ids + test_ad_ids
        sar_dir = "/data/cache_SAR"
        opt_dir = "/data/cache_OPT"

        os.makedirs(sar_dir, exist_ok=True)
        os.makedirs(opt_dir, exist_ok=True)
        wanted_normal_series = {f"{ID}.npy" for ID in set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids)}
        print(f"Estrazione {len(wanted_normal_series)} serie normali", flush=True)
        for modality, cache_dir in [("SAR", sar_dir), ("OPT", opt_dir)]:
            for part_path in sorted(glob(os.path.join(normal_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_normal_series:
                            z.extract(name, cache_dir)

        # embedding moco in tutti i punti selezionati, un solo passaggio
        # ritorna {punto: (q, k)}, q = ottico, k = sar
        def embed(ids):
            dataset = MoCo2encodersLoader(
                    listIDs=ids, sar_map={}, opt_map={}, transform=None,
                    patch_size=patch_size,
                    n_images1=n_images1, n_channels1=n_channels1,
                    n_images2=n_images2, n_channels2=n_channels2,
                )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

            out = {punto: ([], []) for punto in punti}
            with torch.no_grad(), torch.cuda.amp.autocast():
                for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                    im_q = im_q.to(device)
                    im_k = im_k.to(device)

                    # encoder: latente autoencoder + pooling di moco
                    zq = model.encoder_q(im_q)
                    zk = model.encoder_k(im_k)
                    fq = {"encoder": model.pool_q(zq).flatten(1)}
                    fk = {"encoder": model.pool_k(zk).flatten(1)}

                    # NUOVO: latente autoencoder con pooling più fine (nessun peso coinvolto)
                    if do_encoder_fine:
                        fq["encoder_fine"] = nn.functional.adaptive_avg_pool2d(zq, pool_fine).flatten(1)
                        fk["encoder_fine"] = nn.functional.adaptive_avg_pool2d(zk, pool_fine).flatten(1)

                    # pre_proj: conv + pooling (come contrastive_loss); la mappa conv resta per i punti successivi
                    if usa_conv:
                        cq = model.conv_q(zq)
                        ck = model.conv_k(zk)
                    else:
                        cq, ck = zq, zk
                    hq = model.pool_q(cq).flatten(1)
                    hk = model.pool_k(ck).flatten(1)
                    fq["pre_proj"], fk["pre_proj"] = hq, hk

                    # NUOVO: mappa conv con pooling più fine (solo estrazione: il proiettore non la vede)
                    if do_pre_proj_fine:
                        fq["pre_proj_fine"] = nn.functional.adaptive_avg_pool2d(cq, pool_fine).flatten(1)
                        fk["pre_proj_fine"] = nn.functional.adaptive_avg_pool2d(ck, pool_fine).flatten(1)

                    # pre_proj normalizzato e spostato rispetto al centro (spazio della loss di ms_pre)
                    if moco_ms_pre:
                        fq["pre_proj_ms"] = model.mean_shift(nn.functional.normalize(hq.float(), dim=1), model.center_q)
                        fk["pre_proj_ms"] = model.mean_shift(nn.functional.normalize(hk.float(), dim=1), model.center_k)

                    # NUOVO: ramo denso -> un vettore normalizzato per cella, poi pooling di moco
                    if do_dense:
                        dq = nn.functional.normalize(model.dense_q(cq).float(), dim=1)
                        dk = nn.functional.normalize(model.dense_k(ck).float(), dim=1)
                        fq["dense"] = model.pool_q(dq).flatten(1)
                        fk["dense"] = model.pool_k(dk).flatten(1)

                    # proj_raw: proiettore senza normalizzazione
                    pq = model.proj_q(hq)
                    pk = model.proj_k(hk)
                    fq["proj_raw"], fk["proj_raw"] = pq, pk

                    # post_proj: proiettore + normalizzazione (+ spostamento per il modello ms)
                    if moco_ms:
                        fq["post_proj"] = model.mean_shift(nn.functional.normalize(pq, dim=1), model.center_q)
                        fk["post_proj"] = model.mean_shift(nn.functional.normalize(pk, dim=1), model.center_k)
                    else:
                        fq["post_proj"] = nn.functional.normalize(pq, dim=1)
                        fk["post_proj"] = nn.functional.normalize(pk, dim=1)

                    for punto in punti:
                        out[punto][0].append(fq[punto].float().flatten(1).cpu())
                        out[punto][1].append(fk[punto].float().flatten(1).cpu())
            return {punto: (torch.cat(q).numpy(), torch.cat(k).numpy()) for punto, (q, k) in out.items()}

        def print_dim(nome, emb):
            dims = " | ".join(f"{punto} q {q.shape[1]} + k {k.shape[1]}" for punto, (q, k) in emb.items())
            print(f"OK -> {nome}: {len(next(iter(emb.values()))[0])} serie | {dims}", "\n", flush=True)

        # embedding serie di fit
        print("Embedding MoCo: serie di fit ocsvm", flush=True)
        emb = embed(normal_fit_ids)
        for punto in punti: E[punto]["fit"] = emb[punto]
        print_dim("normali fit", emb)

        # embedding serie thr
        print("Embedding MoCo: serie thr ocsvm", flush=True)
        emb = embed(normal_thr_ids)
        for punto in punti: E[punto]["thr"] = emb[punto]
        print_dim("normali thr", emb)

        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        emb = embed(normal_test_ids)
        for punto in punti: E[punto]["normal"] = emb[punto]
        print_dim("normali test", emb)

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
        emb = embed(anom_ids)
        for punto in punti: E[punto]["anom"] = emb[punto]
        print_dim("anomale", emb)

    # cartella radice, una sottocartella per punto
    out_root = f"/data/anomaly_results_ocsvm_{job_name}"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root)
    riepilogo = {}

    # anomaly detection ripetuta per ogni punto
    for punto in punti:
        print(f"===== {punto} =====", flush=True)
        q_fit, k_fit = E[punto]["fit"]
        q_thr, k_thr = E[punto]["thr"]
        q_normal, k_normal = E[punto]["normal"]
        q_anom, k_anom = E[punto]["anom"]

        # concatenazione
        w_fit = np.concatenate([q_fit, k_fit], axis=1).astype(np.float64)
        w_thr = np.concatenate([q_thr, k_thr], axis=1).astype(np.float64)
        w_normal = np.concatenate([q_normal, k_normal], axis=1).astype(np.float64)
        w_anom = np.concatenate([q_anom, k_anom],axis=1).astype(np.float64)

        if standardize:
            scaler = StandardScaler().fit(w_fit)
            w_fit, w_thr, w_normal, w_anom = scaler.transform(w_fit), scaler.transform(w_thr), scaler.transform(w_normal), scaler.transform(w_anom)

        if whitening == "lw":
            lw = LedoitWolf().fit(w_fit)
            lam, V = np.linalg.eigh(lw.covariance_)
            W = V / np.sqrt(lam)
            z_fit = (w_fit - lw.location_) @ W
            z_thr = (w_thr - lw.location_) @ W
            z_normal = (w_normal - lw.location_) @ W
            z_anom = (w_anom - lw.location_) @ W
            n_componenti = w_fit.shape[1]
            shrinkage = float(lw.shrinkage_)
            print(f"Ledoit-Wolf: {n_componenti} dimensioni | shrinkage {shrinkage:.3f} | "
                  f"autovalore min {lam.min():.4f} max {lam.max():.2f}", flush=True)
        else:
            pw = PCA(n_components=pca_punto[punto], whiten=True).fit(w_fit)
            z_fit = pw.transform(w_fit)
            z_thr = pw.transform(w_thr)
            z_normal = pw.transform(w_normal)
            z_anom = pw.transform(w_anom)
            n_componenti = int(pw.n_components_)
            shrinkage = None
            print(f"PCA: {n_componenti} componenti su {w_fit.shape[1]} dimensioni", flush=True)

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
        out_dir = os.path.join(out_root, punto)
        os.makedirs(out_dir)

        # punteggi
        df = pd.DataFrame({
            "ID": normal_test_ids + anom_ids,
            "label": [0] * len(score_normal) + [1] * len(score_anom),
            "score": np.concatenate([score_normal, score_anom]),
        })
        df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
        pd.DataFrame({"ID": normal_thr_ids, "score": score_thr}).to_csv(os.path.join(out_dir, "scores_thr.csv"), index=False)

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
        print(flush=True)

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
            "feature_point": punto, "pca_components": pca_punto[punto], "n_componenti": n_componenti,
            "whitening": whitening, "shrinkage": shrinkage, "fit_on_train": fit_on_train,
            "weights_version": weights_version or "ultima",
            "pool_grid": pool_grid, "pool_fine": pool_fine if punto.endswith("_fine") else None,   # NUOVO
            "ocsvm": {"nu": ocsvm_nu, "gamma": gamma, "dim": int(z_fit.shape[1]), "frac_outside_fit": frac_outside_fit},
            "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids), "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids),
            "auroc": float(auroc),
            "rules": rules,
        }
        with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        riepilogo[punto] = {"auroc": float(auroc), "n_componenti": n_componenti, "shrinkage": shrinkage}

    # riepilogo di tutti i punti (serve anche per old_vectors e per ad_knn)
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump({"job_name": job_name, "moco_version": moco_version, "dataset_anomalies": dataset_anomalies,
                   "fit_on_train": fit_on_train, "whitening": whitening,
                   "weights_version": weights_version or "ultima",
                   "pool_grid": pool_grid, "pool_fine": pool_fine,                                        # NUOVO
                   "punti": riepilogo}, f, indent=2)

    print("===== RIEPILOGO =====", flush=True)
    for punto, r in riepilogo.items():
        print(f"{punto:13s}: auroc {r['auroc']:.4f} | {r['n_componenti']} componenti", flush=True)

    shutil.make_archive(out_root, "zip", out_root)
    project_work.log_artifact(name=f"anomaly_results_ocsvm_{job_name}", source=f"{out_root}.zip", kind="artifact")

    return "TERMINATO -> anomaly detection ocsvm"    




@handler()
def ad_knn(
    job_name: str = "knn_test",
    vectors_from: str = "",                         
    punti: str = "encoder, pre_proj",                
    distanze: str = "coseno, lw",                    
    ks: str = "1,5,20",                            
    blocco: int = 1000,                             
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    lista_punti = [p.strip() for p in punti.split(",") if p.strip()]
    lista_distanze = [d.strip() for d in distanze.split(",") if d.strip()]
    lista_k = sorted({int(k) for k in ks.split(",") if k.strip()})
    k_max = max(lista_k)
    for d in lista_distanze:
        if d not in ("coseno", "lw"):
            raise ValueError(f"distanza non valida: {d}")
    if not vectors_from:
        raise ValueError("indica in vectors_from il job_name di ad_ocsvm3 con gli embedding")

    project_work = dh.get_project("floods")

    # ---------- embedding dal job di ad_ocsvm3 ----------
    print(f"Embedding da: anomaly_results_ocsvm_{vectors_from}", flush=True)
    p = project_work.get_artifact(f"anomaly_results_ocsvm_{vectors_from}").download("/data/vectors.zip", overwrite=True)
    with zipfile.ZipFile(p) as z:
        summary_vecchio = json.loads(z.read("summary.json"))
        mancanti = [pt for pt in lista_punti if f"{pt}/embeddings.npz" not in z.namelist()]
        if mancanti:
            raise ValueError(f"punti non presenti in {vectors_from}: {mancanti}")
        for pt in lista_punti:
            z.extract(f"{pt}/embeddings.npz", "/data/vectors")
    print(f"OK -> moco {summary_vecchio['moco_version']} | punti: {', '.join(lista_punti)}", "\n", flush=True)

    def carica(punto):
        E = np.load(f"/data/vectors/{punto}/embeddings.npz")
        w = {g: np.concatenate([E[f"{g}_q"], E[f"{g}_k"]], axis=1).astype(np.float32)
             for g in ("fit", "thr", "normal", "anom")}
        ids = {g: [str(i) for i in E[f"{g}_ids"]] for g in ("thr", "normal", "anom")}
        return w, ids

    def prepara(w, modo):
        if modo == "coseno":
            # ogni vettore sulla sfera: la distanza diventa 1 - similarità coseno
            return {g: v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12) for g, v in w.items()}, None
        # lw: standardizzazione + sbiancamento con la covarianza di Ledoit-Wolf
        mu, sd = w["fit"].mean(0), w["fit"].std(0) + 1e-8
        w = {g: (v - mu) / sd for g, v in w.items()}
        X = w["fit"]
        alpha = ledoit_wolf_shrinkage(X)
        lam, V = np.linalg.eigh((X.T @ X).astype(np.float64) / len(X))
        W = (V / np.sqrt((1 - alpha) * lam + alpha * lam.mean())).astype(np.float32)
        return {g: v @ W for g, v in w.items()}, float(alpha)

    def vicini(test, F, modo):
        """Le k_max distanze più piccole di ogni serie di test dalle serie di fit, in ordine crescente."""
        out = []
        for i in range(0, len(test), blocco):
            T = torch.from_numpy(test[i:i + blocco]).to(device)
            if modo == "coseno":
                d = 1 - T @ F.T
            else:
                d = torch.cdist(T, F)
            out.append(d.topk(k_max, dim=1, largest=False).values.cpu().numpy())   
        return np.concatenate(out)

    out_root = f"/data/anomaly_results_knn_{job_name}"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root)

    false_allarms = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    righe = []

    for punto in lista_punti:
        w, ids = carica(punto)
        print(f"===== {punto} | fit {w['fit'].shape} =====", flush=True)

        for modo in lista_distanze:
            f, shrinkage = prepara(w, modo)
            F = torch.from_numpy(f["fit"]).to(device)                       

            d_thr = vicini(f["thr"], F, modo)
            d_norm = vicini(f["normal"], F, modo)
            d_anom = vicini(f["anom"], F, modo)
            del F
            if device.type == "cuda":
                torch.cuda.empty_cache()

            out_dir = os.path.join(out_root, punto, modo)
            os.makedirs(out_dir)
            df_scores = pd.DataFrame({"ID": ids["normal"] + ids["anom"],
                                      "label": [0] * len(d_norm) + [1] * len(d_anom)})
            df_thr = pd.DataFrame({"ID": ids["thr"]})

            for k in lista_k:
                # score = distanza media dai k vicini (più alta = più anomala)
                s_thr = d_thr[:, :k].mean(1)
                s_norm = d_norm[:, :k].mean(1)
                s_anom = d_anom[:, :k].mean(1)
                y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_anom))]
                auroc = roc_auc_score(y, np.r_[s_norm, s_anom])

                df_scores[f"score_k{k}"] = np.r_[s_norm, s_anom]
                df_thr[f"score_k{k}"] = s_thr

                # soglie dalle normali di soglia, come in ad_ocsvm3 (qui: anomalia = score sopra la soglia)
                rules = {}
                for fa in false_allarms:
                    thr = float(np.quantile(s_thr, 1 - fa))                   
                    tp = int((s_anom > thr).sum()); fn = len(s_anom) - tp
                    fp = int((s_norm > thr).sum()); tn = len(s_norm) - fp
                    prec = tp / (tp + fp) if tp + fp else 0.0
                    rec = tp / (tp + fn)
                    rules[f"FA {fa:.0%}"] = {
                        "soglia": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                        "FA_rate": fp / (fp + tn), "MA_rate": fn / (fn + tp),                   
                        "REC": rec, "PREC": prec,
                        "F1": 2 * prec * rec / (prec + rec) if tp > 0 else 0.0,
                    }

                righe.append({"punto": punto, "distanza": modo, "k": k, "AUROC": float(auroc), "shrinkage": shrinkage})
                with open(os.path.join(out_dir, f"risultati_k{k}.json"), "w") as fp_json:
                    json.dump({"punto": punto, "distanza": modo, "k": k, "auroc": float(auroc),
                               "shrinkage": shrinkage, "rules": rules}, fp_json, indent=2)
                print(f"{punto:11s} | {modo:6s} | k={k:3d} | AUROC {auroc:.4f} | "
                      f"REC a FA 10% {rules['FA 10%']['REC']:.3f}", flush=True)

            df_scores.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
            df_thr.to_csv(os.path.join(out_dir, "scores_thr.csv"), index=False)
        print(flush=True)

    tabella = pd.DataFrame(righe)
    tabella.to_csv(os.path.join(out_root, "tabella.csv"), index=False)
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump({"job_name": job_name, "vectors_from": vectors_from,
                   "moco_version": summary_vecchio["moco_version"], "risultati": righe}, f, indent=2)

    print("===== RIEPILOGO (AUROC) =====", flush=True)
    print(tabella.pivot_table(index=["punto", "distanza"], columns="k", values="AUROC").round(3).to_string(), flush=True)

    shutil.make_archive(out_root, "zip", out_root)
    project_work.log_artifact(name=f"anomaly_results_knn_{job_name}", source=f"{out_root}.zip", kind="artifact")
    return "TERMINATO -> anomaly detection knn"


@handler()
def estrai_vettori(
    job_name: str = "nome_job",
    dataset_standard: str = "Standard",
    dataset_anomalies: str = "test",
    moco_version: str = "check_v1",
    simple_proj: bool = False,
    hidden_channels_dim: int = 64,
    pool_grid: int = 1,
    moco_dim: int = 128,
    moco_k: int = 4096,
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    batch_size: int = 32,
    workers: int = 0,
    min_flood_area: float = 10.0,
    n_series_thr: int = 500,                        # usato solo con fit_on_train=False
    moco_attn: bool = False,
    moco_bot: bool = False,
    moco_ms: bool = False,
    moco_ms_pre: bool = False,
    moco_dense: bool = False,

    do_encoder: bool = True,                        # latente autoencoder + pooling di moco
    do_encoder_fine: bool = False,                  # latente autoencoder + pooling pool_fine
    do_pre_proj: bool = True,                       # dopo conv + pooling, prima del proiettore
    do_pre_proj_fine: bool = False,                 # dopo conv + pooling pool_fine
    do_pre_proj_ms: bool = False,                   # pre_proj normalizzato e spostato rispetto al centro (solo ms_pre)
    do_dense: bool = False,                         # ramo denso normalizzato per cella + pooling di moco (solo dense)
    do_proj_raw: bool = False,                      # dopo il proiettore, prima della normalizzazione
    do_post_proj: bool = False,                     # dopo la normalizzazione (come contrastive_loss)

    pool_fine: int = 8,                             # griglia del pooling dei punti *_fine (solo estrazione)
    fit_on_train: bool = True,                      # True: serie di fit = tutte le serie di train di moco
    max_fit_series: int = 0,                        # 0 = tutte; altrimenti campione casuale
    weights_version: str = "",
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    punti = [p for p, do in (("encoder", do_encoder), ("encoder_fine", do_encoder_fine),
                             ("pre_proj", do_pre_proj), ("pre_proj_fine", do_pre_proj_fine),
                             ("pre_proj_ms", do_pre_proj_ms), ("dense", do_dense),
                             ("proj_raw", do_proj_raw), ("post_proj", do_post_proj)) if do]
    if not punti:
        raise ValueError("nessun punto selezionato: attiva almeno un do_*")
    if do_pre_proj_ms and not moco_ms_pre:
        raise ValueError("do_pre_proj_ms richiede moco_ms_pre=True")
    if do_dense and not moco_dense:
        raise ValueError("do_dense richiede moco_dense=True")
    if sum([moco_attn, moco_bot, moco_ms, moco_ms_pre, moco_dense]) > 1:
        raise ValueError("indica al massimo uno tra moco_attn, moco_bot, moco_ms, moco_ms_pre, moco_dense")
    if (do_encoder_fine or do_pre_proj_fine) and pool_fine < 1:
        raise ValueError("pool_fine deve essere >= 1")
    print(f"punti: {', '.join(punti)} | fit_on_train: {fit_on_train} | pool_grid: {pool_grid} | pool_fine: {pool_fine}",
          "\n", flush=True)

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    # ---------- modello: prima dei dataset, così se fallisce il job si ferma subito ----------
    try:
        def scarica_pesi(nome_artifact):
            if weights_version:
                art = dh.get_artifact(nome_artifact, project="floods", entity_id=weights_version)
            else:
                art = project_work.get_artifact(nome_artifact)
            print(f"pesi: {nome_artifact} (versione {weights_version or 'ultima'})", flush=True)
            return art.download("/data/moco_weights.pth", overwrite=True)

        if moco_attn:
            encoderSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)
            encoderOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=True, spac_net=False).to(device)
            model = MoCo2encoders_attn(
                base_encoder_q=encoderOPT.encoder, base_encoder_k=encoderSAR.encoder,
                dim=moco_dim, K=moco_k, m=moco_m, T=moco_t, symmetric=False,
                hidden_channels_dim=hidden_channels_dim, pool_grid=pool_grid, simple_proj=False,
            ).to(device)
            w_path = scarica_pesi(f"moco_attn_weights_{moco_version}")

        else:
            encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
            encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)

            if moco_bot:
                model = MoCo2encoders_Bot(
                    base_encoder_q=encoderOPT.encoder, base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t, symmetric=False, device=device,
                    hidden_channels_dim=hidden_channels_dim, pool_grid=pool_grid, simple_proj=simple_proj,
                ).to(device)
                w_path = scarica_pesi(f"moco_bot_weights_{moco_version}")
            elif moco_ms:
                model = MoCo2encoders_ms(
                    base_encoder_q=encoderOPT.encoder, base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    hidden_channels_dim=hidden_channels_dim, pool_grid=pool_grid,
                ).to(device)
                w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")
            elif moco_ms_pre:
                model = MoCo2encoders_ms_pre(
                    base_encoder_q=encoderOPT.encoder, base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    hidden_channels_dim=hidden_channels_dim, pool_grid=pool_grid,
                ).to(device)
                w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")
            elif moco_dense:
                # usa gli stessi argomenti del costruttore con cui è stata allenata quella versione
                model = MoCo2encoders_dense(
                    base_encoder_q=encoderOPT.encoder, base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
                    hidden_channels_dim=hidden_channels_dim, pool_grid=pool_grid,
                ).to(device)
                w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")
            else:
                model = MoCo2encoders_2d(
                    base_encoder_q=encoderOPT.encoder, base_encoder_k=encoderSAR.encoder,
                    dim=moco_dim, K=moco_k, m=moco_m, T=moco_t, symmetric=False, device=device,
                    hidden_channels_dim=hidden_channels_dim, simple_proj=simple_proj, pool_grid=pool_grid,
                ).to(device)
                w_path = scarica_pesi(f"moco_2D_weights_{moco_version}")

        print(f"classe: {type(model).__name__}", flush=True)

        checkpoint = torch.load(w_path, map_location=device)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        mancanti, inattesi = model.load_state_dict(state_dict, strict=False)
        if mancanti or inattesi:
            print(f"chiavi mancanti nel checkpoint: {mancanti}", flush=True)
            print(f"chiavi inattese nel checkpoint: {inattesi}", flush=True)
            raise ValueError(f"il checkpoint di {moco_version} non corrisponde alla classe {type(model).__name__}")

        if moco_ms or moco_ms_pre:
            print(f"centri: |c_q| {model.center_q.norm():.3f} | |c_k| {model.center_k.norm():.3f}", flush=True)
            if model.center_q.norm() == 0 or model.center_k.norm() == 0:
                raise ValueError("centri a zero nel checkpoint: il modello non è stato allenato con la mean-shift")

        model.eval()
        usa_conv = not getattr(model, "simple_proj", False)
        print("OK -> modello MoCo caricato", "\n", flush=True)

    except Exception as e:
        print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
        raise

    # ---------- serie anomale ----------
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

    if min_flood_area > 0:
        try:
            flood_area = {}
            for part_path in sorted(glob(os.path.join(anom_path, "MASK", "part*.zip"))):
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

    # ---------- serie normali: stesse liste e stessi split di ad_ocsvm6 ----------
    print(f"Lettura serie normali: Floods_{dataset_standard}_crop_norm", flush=True)
    normal_path = project_data.get_artifact(f"Floods_{dataset_standard}_crop_norm").download(f"/data/Floods_{dataset_standard}_crop_norm")

    prefisso = "moco_attn" if moco_attn else ("moco_bot" if moco_bot else "moco_2D")
    p = project_work.get_artifact(f"{prefisso}_testIdsList_{moco_version}").download("/data/det_test_ids.json", overwrite=True)
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
    normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series + len(anom_ids))]

    if fit_on_train:
        p_train = project_work.get_artifact(f"{prefisso}_trainIdsList_{moco_version}").download("/data/det_train_ids.json", overwrite=True)
        with open(p_train, 'r') as f:
            train_ids = json.load(f)
        normal_fit_ids = [ID for ID in train_ids if ID in normal_ids]
        if max_fit_series > 0 and len(normal_fit_ids) > max_fit_series:
            random.seed(1)
            normal_fit_ids = random.sample(normal_fit_ids, max_fit_series)
        test_set = set(normal_test_ids)
        normal_thr_ids = [ID for ID in shuffled_test_ids if ID not in test_set]
    else:
        normal_fit_ids = shuffled_test_ids[:num_fit_series - n_series_thr]
        normal_thr_ids = shuffled_test_ids[num_fit_series - n_series_thr:num_fit_series]

    assert not (set(normal_fit_ids) & set(normal_thr_ids)), "serie in comune tra fit e soglia"
    assert not (set(normal_fit_ids) & set(normal_test_ids)), "serie in comune tra fit e test"
    assert not (set(normal_thr_ids) & set(normal_test_ids)), "serie in comune tra soglia e test"
    print(f"OK -> {len(normal_fit_ids)} serie fit, {len(normal_thr_ids)} serie thr, {len(normal_test_ids)} serie test normali",
          "\n", flush=True)

    # ---------- estrazione ----------
    sar_dir, opt_dir = "/data/cache_SAR", "/data/cache_OPT"

    def riempi_cache(src_path, ids):
        for d in (sar_dir, opt_dir):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        wanted = {f"{ID}.npy" for ID in ids}
        print(f"Estrazione {len(wanted)} serie", flush=True)
        for modality, cache_dir in (("SAR", sar_dir), ("OPT", opt_dir)):
            for part_path in sorted(glob(os.path.join(src_path, modality, "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted:
                            z.extract(name, cache_dir)

    def embed(ids):
        dataset = MoCo2encodersLoader(
            listIDs=ids, sar_map={}, opt_map={}, transform=None, patch_size=patch_size,
            n_images1=n_images1, n_channels1=n_channels1, n_images2=n_images2, n_channels2=n_channels2,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

        out = {punto: ([], []) for punto in punti}
        with torch.no_grad(), torch.cuda.amp.autocast():
            for im_q, im_k in tqdm(loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
                im_q, im_k = im_q.to(device), im_k.to(device)

                zq, zk = model.encoder_q(im_q), model.encoder_k(im_k)
                fq = {"encoder": model.pool_q(zq).flatten(1)}
                fk = {"encoder": model.pool_k(zk).flatten(1)}

                if do_encoder_fine:
                    fq["encoder_fine"] = nn.functional.adaptive_avg_pool2d(zq, pool_fine).flatten(1)
                    fk["encoder_fine"] = nn.functional.adaptive_avg_pool2d(zk, pool_fine).flatten(1)

                cq, ck = (model.conv_q(zq), model.conv_k(zk)) if usa_conv else (zq, zk)
                hq, hk = model.pool_q(cq).flatten(1), model.pool_k(ck).flatten(1)
                fq["pre_proj"], fk["pre_proj"] = hq, hk

                if do_pre_proj_fine:
                    fq["pre_proj_fine"] = nn.functional.adaptive_avg_pool2d(cq, pool_fine).flatten(1)
                    fk["pre_proj_fine"] = nn.functional.adaptive_avg_pool2d(ck, pool_fine).flatten(1)

                if do_pre_proj_ms:
                    fq["pre_proj_ms"] = model.mean_shift(nn.functional.normalize(hq.float(), dim=1), model.center_q)
                    fk["pre_proj_ms"] = model.mean_shift(nn.functional.normalize(hk.float(), dim=1), model.center_k)

                if do_dense:
                    dq = nn.functional.normalize(model.dense_q(cq).float(), dim=1)
                    dk = nn.functional.normalize(model.dense_k(ck).float(), dim=1)
                    fq["dense"], fk["dense"] = model.pool_q(dq).flatten(1), model.pool_k(dk).flatten(1)

                if do_proj_raw or do_post_proj:                     # il proiettore solo se serve
                    pq, pk = model.proj_q(hq), model.proj_k(hk)
                    fq["proj_raw"], fk["proj_raw"] = pq, pk
                    if moco_ms:
                        fq["post_proj"] = model.mean_shift(nn.functional.normalize(pq, dim=1), model.center_q)
                        fk["post_proj"] = model.mean_shift(nn.functional.normalize(pk, dim=1), model.center_k)
                    else:
                        fq["post_proj"] = nn.functional.normalize(pq, dim=1)
                        fk["post_proj"] = nn.functional.normalize(pk, dim=1)

                for punto in punti:
                    out[punto][0].append(fq[punto].float().flatten(1).cpu())
                    out[punto][1].append(fk[punto].float().flatten(1).cpu())
        return {punto: (torch.cat(q).numpy(), torch.cat(k).numpy()) for punto, (q, k) in out.items()}

    E = {punto: {} for punto in punti}
    gruppi = (("fit", normal_path, normal_fit_ids), ("thr", normal_path, normal_thr_ids),
              ("normal", normal_path, normal_test_ids), ("anom", anom_path, anom_ids))

    riempi_cache(normal_path, set(normal_fit_ids) | set(normal_thr_ids) | set(normal_test_ids))
    for gruppo, src_path, ids in gruppi:
        if gruppo == "anom":
            riempi_cache(anom_path, anom_ids)
        print(f"Embedding: {gruppo} ({len(ids)} serie)", flush=True)
        emb = embed(ids)
        for punto in punti:
            E[punto][gruppo] = emb[punto]
        dims = " | ".join(f"{punto} {q.shape[1]}+{k.shape[1]}" for punto, (q, k) in emb.items())
        print(f"OK -> {gruppo}: {dims}", "\n", flush=True)

    # ---------- salvataggio: stessa struttura di ad_ocsvm6, leggibile da ad_knn e come old_vectors ----------
    out_root = f"/data/anomaly_results_ocsvm_{job_name}"
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root)

    for punto in punti:
        out_dir = os.path.join(out_root, punto)
        os.makedirs(out_dir)
        (q_fit, k_fit), (q_thr, k_thr) = E[punto]["fit"], E[punto]["thr"]
        (q_normal, k_normal), (q_anom, k_anom) = E[punto]["normal"], E[punto]["anom"]
        np.savez_compressed(
            os.path.join(out_dir, "embeddings.npz"),
            fit_ids=np.array(normal_fit_ids), fit_q=q_fit, fit_k=k_fit,
            thr_ids=np.array(normal_thr_ids), thr_q=q_thr, thr_k=k_thr,
            normal_ids=np.array(normal_test_ids), normal_q=q_normal, normal_k=k_normal,
            anom_ids=np.array(anom_ids), anom_q=q_anom, anom_k=k_anom,
        )
        # file minimo richiesto da ad_ocsvm6 quando si usa questo job come old_vectors
        with open(os.path.join(out_dir, "anomaly_results.json"), "w") as f:
            json.dump({"job_name": job_name, "metodo": "solo_embedding", "feature_point": punto,
                       "moco_version": moco_version, "dataset_anomalies": dataset_anomalies,
                       "min_flood_area": min_flood_area, "weights_version": weights_version or "ultima",
                       "pool_grid": pool_grid, "pool_fine": pool_fine if punto.endswith("_fine") else None,
                       "dim": int(q_fit.shape[1] + k_fit.shape[1])}, f, indent=2)

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump({"job_name": job_name, "metodo": "solo_embedding", "moco_version": moco_version,
                   "dataset_anomalies": dataset_anomalies, "fit_on_train": fit_on_train,
                   "weights_version": weights_version or "ultima",
                   "pool_grid": pool_grid, "pool_fine": pool_fine, "punti": punti,
                   "n_fit": len(normal_fit_ids), "n_thr": len(normal_thr_ids),
                   "n_normali": len(normal_test_ids), "n_anomale": len(anom_ids)}, f, indent=2)

    shutil.make_archive(out_root, "zip", out_root)
    project_work.log_artifact(name=f"anomaly_results_ocsvm_{job_name}", source=f"{out_root}.zip", kind="artifact")
    print(f"OK -> embedding salvati: anomaly_results_ocsvm_{job_name} | punti: {', '.join(punti)}", flush=True)
    return "TERMINATO -> estrazione embedding"








# ============================================================================
# DIAGNOSTICA DEGLI EMBEDDING
# Legge gli artifact anomaly_results_ocsvm_<job> prodotti da ad_ocsvm6 /
# estrai_vettori e confronta più combinazioni (job) e più punti di estrazione.
# Non ricalcola niente con il modello: lavora solo sui vettori già salvati.
#
# Output (una cartella, caricata come artifact):
#   tabella.csv            una riga per (job, punto) con tutte le metriche
#   cka.csv / cka.png      similarità tra rappresentazioni (CKA lineare)
#   spettro.png            varianza spiegata cumulativa (collasso dimensionale)
#   auroc_per_dim.png      quanto segnale porta ogni singola dimensione
#   <job>__<punto>.png     PCA 2D + istogrammi dei punteggi
# ============================================================================
import os, json, shutil, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # nessuno schermo sul cluster: salva solo su file
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

GRUPPI = ("fit", "thr", "normal", "anom")


# ---------------------------------------------------------------- caricamento
def carica_npz(path):
    E = np.load(path)
    v = {g: (E[f"{g}_q"].astype(np.float64), E[f"{g}_k"].astype(np.float64)) for g in GRUPPI}
    ids = {g: [str(i) for i in E[f"{g}_ids"]] for g in GRUPPI}
    flood = E["anom_flood"] if "anom_flood" in E.files else None    # opzionale (vedi nota in fondo)
    return v, ids, flood


# ---------------------------------------------------------------- metriche
def normalizza(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def knn_score(test, fit, k=5, blocco=2000):
    """Distanza coseno media dai k vicini di fit (come ad_knn, modo 'coseno')."""
    test, fit = normalizza(test), normalizza(fit)
    out = []
    for i in range(0, len(test), blocco):
        d = 1 - test[i:i + blocco] @ fit.T
        out.append(np.sort(np.partition(d, k - 1, axis=1)[:, :k], axis=1).mean(1))
    return np.concatenate(out)


def auroc(s_norm, s_anom):
    y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_anom))]
    return roc_auc_score(y, np.r_[s_norm, s_anom])


def spettro(X):
    """Varianza spiegata cumulativa, rango effettivo e numero di condizionamento."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)            # stessa standardizzazione di ad_ocsvm6
    lam = np.clip(np.linalg.eigvalsh(np.cov(Xs.T))[::-1], 1e-12, None)
    p = lam / lam.sum()
    rango_eff = float(np.exp(-(p * np.log(p)).sum()))   # Roy & Vetterli 2007
    cum = np.cumsum(p)
    return {
        "dim": X.shape[1],
        "rango_eff": rango_eff,
        "rango_eff_rel": rango_eff / X.shape[1],          # confrontabile tra dimensioni diverse (pool_grid)
        "comp_90": int(np.searchsorted(cum, 0.90) + 1),
        "comp_99": int(np.searchsorted(cum, 0.99) + 1),
        "log10_cond": float(np.log10(lam[0] / lam[-1])),
        "shrinkage_lw": float(LedoitWolf().fit(Xs).shrinkage_),
    }, cum


def alignment(q, k):
    """Wang & Isola 2020: distanza media tra le due modalità della stessa serie (basso = allineate)."""
    return float((np.linalg.norm(normalizza(q) - normalizza(k), axis=1) ** 2).mean())


def uniformity(x, n_max=3000, t=2, seed=0):
    """Wang & Isola 2020: quanto i vettori coprono la sfera (più negativo = più uniforme)."""
    rng = np.random.default_rng(seed)
    x = normalizza(x[rng.choice(len(x), min(n_max, len(x)), replace=False)])
    d2 = 2 - 2 * (x @ x.T)
    iu = np.triu_indices(len(x), 1)
    return float(np.log(np.exp(-t * d2[iu]).mean()))


def auroc_per_dim(Xn, Xa):
    """AUROC di ogni singola dimensione presa da sola, ripiegato attorno a 0.5."""
    y = np.r_[np.zeros(len(Xn)), np.ones(len(Xa))]
    X = np.r_[Xn, Xa]
    a = np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
    return np.maximum(a, 1 - a)


def linear_cka(X, Y):
    """Kornblith et al. 2019. X, Y: stesse serie nello stesso ordine."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


# ---------------------------------------------------------------- analisi di una combinazione
def analizza(nome, v, flood, out_dir, k=5):
    (qf, kf), (qn, kn), (qa, ka) = v["fit"], v["normal"], v["anom"]
    cat = lambda q, kk: np.concatenate([q, kk], axis=1)
    riga, curve = {"combinazione": nome}, {}

    # 1) da dove viene il segnale: solo ottico, solo sar, concatenati, accordo tra modalità
    s = {}
    for lab, (F, N, A) in {"q": (qf, qn, qa), "k": (kf, kn, ka),
                           "qk": (cat(qf, kf), cat(qn, kn), cat(qa, ka))}.items():
        s[lab] = (knn_score(N, F, k), knn_score(A, F, k))
        riga[f"auroc_knn_{lab}"] = auroc(*s[lab])
    if qf.shape[1] == kf.shape[1]:                       # stessa dimensione: confronto diretto q-k
        cos_n = (normalizza(qn) * normalizza(kn)).sum(1)
        cos_a = (normalizza(qa) * normalizza(ka)).sum(1)
        riga["cos_qk_normali"], riga["cos_qk_anomale"] = float(cos_n.mean()), float(cos_a.mean())
        riga["auroc_1-cos_qk"] = auroc(1 - cos_n, 1 - cos_a)
    else:
        cos_n = cos_a = None

    # 2) forma dello spazio (sulle serie di fit, come vede l'ocsvm)
    for lab, X in {"q": qf, "k": kf, "qk": cat(qf, kf)}.items():
        info, cum = spettro(X)
        riga.update({f"{lab}_{c}": val for c, val in info.items()})
        curve[lab] = cum

    # 3) alignment / uniformity
    if qf.shape[1] == kf.shape[1]:
        riga["alignment"] = alignment(qf, kf)
    riga["uniformity_q"], riga["uniformity_k"] = uniformity(qf), uniformity(kf)

    # 4) quante dimensioni portano segnale da sole
    apd = auroc_per_dim(cat(qn, kn), cat(qa, ka))
    riga["dim_auroc>0.6"] = int((apd > 0.6).sum())
    riga["frac_dim_auroc>0.6"] = float((apd > 0.6).mean())   # confrontabile tra pool_grid diversi
    riga["dim_auroc_max"] = float(apd.max())
    curve["apd"] = np.sort(apd)[::-1]

    # ---- figura per questa combinazione
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    Wf, Wn, Wa = cat(qf, kf), cat(qn, kn), cat(qa, ka)
    mu, sd = Wf.mean(0), Wf.std(0) + 1e-8
    pca = PCA(2).fit((Wf - mu) / sd)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Wf), min(3000, len(Wf)), replace=False)
    for X, col, lab, a in ((Wf[idx], "lightgray", "fit", .3), (Wn, "tab:blue", "normali test", .5),
                           (Wa, "tab:red", "anomale", .5)):
        P = pca.transform((X - mu) / sd)
        c = col if (lab != "anomale" or flood is None) else None
        sc = ax[0].scatter(P[:, 0], P[:, 1], s=4, alpha=a, c=c if c else flood,
                           cmap=None if c else "Reds", label=lab)
    if flood is not None:
        fig.colorbar(sc, ax=ax[0], label="% area allagata")
    ax[0].set_title(f"PCA q+k  (var. spiegata {pca.explained_variance_ratio_.sum():.0%})")
    ax[0].legend(markerscale=4)

    for lab, col in (("q", "tab:green"), ("k", "tab:purple"), ("qk", "black")):
        bins = np.linspace(min(s[lab][0].min(), s[lab][1].min()), max(s[lab][0].max(), s[lab][1].max()), 60)
        ax[1].hist(s[lab][0], bins, histtype="step", color=col, ls="--")
        ax[1].hist(s[lab][1], bins, histtype="step", color=col,
                   label=f"{lab}  AUROC {riga[f'auroc_knn_{lab}']:.3f}")
    ax[1].set_title(f"score kNN (k={k}) — tratteggio normali, pieno anomale")
    ax[1].legend()

    if cos_n is not None:
        bins = np.linspace(min(cos_n.min(), cos_a.min()), 1, 60)
        ax[2].hist(cos_n, bins, alpha=.5, label="normali", density=True)
        ax[2].hist(cos_a, bins, alpha=.5, label="anomale", density=True, color="tab:red")
        ax[2].set_title(f"cos(ottico, SAR) stessa serie — AUROC {riga['auroc_1-cos_qk']:.3f}")
        ax[2].legend()
    else:
        ax[2].axis("off")

    if flood is not None:                                # score contro dimensione dell'alluvione
        riga["spearman_score_flood"] = float(pd.Series(s["qk"][1]).corr(pd.Series(flood), method="spearman"))

    fig.suptitle(nome)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{nome.replace('/', '__')}.png"), dpi=110)
    plt.close(fig)
    return riga, curve


# ---------------------------------------------------------------- confronto tra combinazioni
def diagnostica(sorgenti, out_dir, k=5):
    """sorgenti: {nome_combinazione: percorso embeddings.npz}"""
    os.makedirs(out_dir, exist_ok=True)
    righe, curve, dati = [], {}, {}
    for nome, path in sorgenti.items():
        v, ids, flood = carica_npz(path)
        print(f"===== {nome} | q {v['fit'][0].shape[1]} + k {v['fit'][1].shape[1]} dim =====", flush=True)
        riga, curve[nome] = analizza(nome, v, flood, out_dir, k)
        righe.append(riga)
        dati[nome] = (v, ids)
        print("  " + " | ".join(f"{c} {riga[c]:.3f}" for c in riga if c.startswith("auroc")), flush=True)

    tab = pd.DataFrame(righe).set_index("combinazione")
    tab.to_csv(os.path.join(out_dir, "tabella.csv"))

    # spettro e auroc per dimensione, tutte le combinazioni sullo stesso grafico
    for chiave, fname, titolo, xl, yl in (
            ("qk", "spettro.png", "varianza spiegata cumulativa (q+k, fit)", "componenti", "frazione"),
            ("apd", "auroc_per_dim.png", "AUROC di ogni dimensione da sola (ordinato)", "dimensione (rank)", "AUROC")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for nome, c in curve.items():
            ax.plot(np.arange(1, len(c[chiave]) + 1), c[chiave], label=nome)
        ax.set_xscale("log"); ax.set_title(titolo); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname), dpi=110); plt.close(fig)

    # CKA sulle serie (normali test + anomale) presenti in tutte le combinazioni
    comuni = None
    for nome, (v, ids) in dati.items():
        s = set(ids["normal"]) | set(ids["anom"])
        comuni = s if comuni is None else comuni & s
    comuni = sorted(comuni)
    if len(comuni) >= 50 and len(dati) > 1:
        M = {}
        for nome, (v, ids) in dati.items():
            pos = {ID: i for i, ID in enumerate(ids["normal"] + ids["anom"])}
            W = np.r_[np.concatenate(v["normal"], 1), np.concatenate(v["anom"], 1)]
            M[nome] = W[[pos[ID] for ID in comuni]]
        nomi = list(M)
        C = np.array([[linear_cka(M[a], M[b]) for b in nomi] for a in nomi])
        pd.DataFrame(C, index=nomi, columns=nomi).to_csv(os.path.join(out_dir, "cka.csv"))
        fig, ax = plt.subplots(figsize=(1.2 * len(nomi) + 3, 1.2 * len(nomi) + 2))
        im = ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(nomi)), nomi, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(nomi)), nomi, fontsize=7)
        for i in range(len(nomi)):
            for j in range(len(nomi)):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", color="w" if C[i, j] < .6 else "k", fontsize=7)
        fig.colorbar(im); ax.set_title(f"CKA lineare su {len(comuni)} serie comuni")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "cka.png"), dpi=110); plt.close(fig)
    else:
        print(f"CKA saltata: {len(comuni)} serie in comune", flush=True)

    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("\n===== TABELLA =====\n", tab.round(3).T.to_string(), flush=True)
    return tab


# ---------------------------------------------------------------- handler DigitalHub
@handler()
def diagnostica_vettori(
    job_name: str = "diag_test",
    vectors_from: str = "",          # job_name separati da virgola (artifact anomaly_results_ocsvm_<job>)
    punti: str = "encoder,pre_proj",
    k: int = 5,
):
    project_work = dh.get_project("floods")
    sorgenti = {}
    for job in [j.strip() for j in vectors_from.split(",") if j.strip()]:
        p = project_work.get_artifact(f"anomaly_results_ocsvm_{job}").download(f"/data/vec_{job}.zip", overwrite=True)
        with zipfile.ZipFile(p) as z:
            summ = json.loads(z.read("summary.json")) if "summary.json" in z.namelist() else {}
            pg = summ.get("pool_grid", "?")
            for pt in [x.strip() for x in punti.split(",") if x.strip()]:
                if f"{pt}/embeddings.npz" in z.namelist():
                    z.extract(f"{pt}/embeddings.npz", f"/data/vec_{job}")
                    # pool_grid nel nome: i punti *_fine usano invece pool_fine
                    g = summ.get("pool_fine", "?") if pt.endswith("_fine") else pg
                    sorgenti[f"{job}/{pt} g{g}"] = f"/data/vec_{job}/{pt}/embeddings.npz"
                else:
                    print(f"manca {pt} in {job}", flush=True)
        os.remove(p)                             # zip completo non più necessario
    if not sorgenti:
        raise ValueError("nessun embedding trovato")

    out_root = f"/data/diagnostica_{job_name}"
    shutil.rmtree(out_root, ignore_errors=True)
    diagnostica(sorgenti, out_root, k)
    shutil.make_archive(out_root, "zip", out_root)
    project_work.log_artifact(name=f"diagnostica_{job_name}", source=f"{out_root}.zip", kind="artifact")
    return "TERMINATO -> diagnostica"

# NOTA: per avere il colore per % di area allagata e la correlazione score/alluvione,
# in estrai_vettori / ad_ocsvm6 aggiungi a np.savez_compressed:
#     anom_flood=np.array([flood_area.get(ID, np.nan) for ID in anom_ids]),




# ============================================================================
# ANALISI SPETTRALE DEGLI EMBEDDING (tutte le serie di fit, tutte le componenti)
#   1) PCA del concatenato [ottico | SAR]: varianza, quota ottica e AUROC per componente
#   2) componenti minori: AUROC scartando le M componenti piu' grandi (OPT, SAR, OPT+SAR)
#   3) kNN lw: completa con scomposizione ottica / SAR / incrociata, a blocchi, singole modalita'
# Output: artifact spettrale_<job_name> (zip con CSV e PNG)
# ============================================================================
import io, os, json, shutil, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.metrics import roc_auc_score


def _auroc(sn, sa):
    return roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sa))], np.r_[sn, sa])


def _auroc_colonne(Pn, Pa):
    """AUROC (ripiegato attorno a 0.5) di ogni colonna, con i ranghi: tutte le colonne in una volta."""
    r = rankdata(np.r_[Pn, Pa], axis=0)
    n_n, n_a = len(Pn), len(Pa)
    a = (r[n_n:].sum(0) - n_a * (n_a + 1) / 2) / (n_n * n_a)
    return np.maximum(a, 1 - a)


def _spettro(Xf, max_dim_shrink, shrink_fallback):
    """Standardizzazione sulle serie di fit + SVD completa (tutte le componenti stimabili)."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
    Z = (Xf - mu) / sd
    c = Z.mean(0)
    _, S, Vt = np.linalg.svd(Z - c, full_matrices=False)          # Vt: (r, D), r = min(n_fit, D)
    lam = (S.astype(np.float64) ** 2) / len(Z)                     # varianza di ogni componente
    if Z.shape[1] <= max_dim_shrink:
        alpha = float(ledoit_wolf_shrinkage(Z.astype(np.float64)))
    else:
        alpha = shrink_fallback                                    # D x D troppo pesante: shrinkage fisso
    return {"mu": mu, "sd": sd, "c": c, "Vt": Vt, "lam": lam, "alpha": alpha,
            "completo": Vt.shape[0] == Z.shape[1]}                 # completo = tutte le D direzioni stimate


def _coordinate(sp, X):
    return (((X - sp["mu"]) / sp["sd"]) - sp["c"]) @ sp["Vt"].T     # (N, r)


def _curva_minori(sp, Xn, Xa, tagli):
    """Score = sum_{j >= M} coord_j^2 / lam_j (lam con shrinkage): distanza di Mahalanobis dalla media
    delle normali usando solo le componenti dalla M+1 in poi. Ritorna {M: AUROC}."""
    lam_s = (1 - sp["alpha"]) * sp["lam"] + sp["alpha"] * sp["lam"].mean()
    code = {}
    for nome, X in (("n", Xn), ("a", Xa)):
        contr = _coordinate(sp, X) ** 2 / lam_s
        code[nome] = np.cumsum(contr[:, ::-1], axis=1)[:, ::-1]    # code[:, M] = somma dalla componente M in poi
    return {M: _auroc(code["n"][:, M], code["a"][:, M]) for M in tagli if M < len(lam_s)}


def _lw_da_spettro(sp):
    """Sbiancamento W e precisione P della covarianza Ledoit-Wolf, riusando la SVD (nessuna eigh in piu')."""
    lam = (1 - sp["alpha"]) * sp["lam"] + sp["alpha"] * sp["lam"].mean()
    V = sp["Vt"].T.astype(np.float64)
    return V / np.sqrt(lam), (V / lam) @ V.T


def _knn_lw(Xt, Xf, sp, W, P=None, Dq=None, k=5, blocco=500):
    """score = d^2 medio dai K vicini (Mahalanobis lw); con P e Dq scompone in ottica + SAR + incrociata."""
    Zt = (((Xt - sp["mu"]) / sp["sd"]) - sp["c"]).astype(np.float64)
    Zf = (((Xf - sp["mu"]) / sp["sd"]) - sp["c"]).astype(np.float64)
    Yf = Zf @ W; nYf = (Yf ** 2).sum(1)
    score, opt, sar, inc = [], [], [], []
    for i in range(0, len(Zt), blocco):
        Zb = Zt[i:i + blocco]; Yb = Zb @ W
        d = np.clip((Yb ** 2).sum(1)[:, None] + nYf[None] - 2 * Yb @ Yf.T, 0, None)
        vic = np.argpartition(d, k - 1, axis=1)[:, :k]
        score.append(np.take_along_axis(d, vic, 1).mean(1))
        if P is not None:
            Delta = Zb[:, None, :] - Zf[vic]
            Do, Ds = Delta[..., :Dq], Delta[..., Dq:]
            opt.append(((Do @ P[:Dq, :Dq]) * Do).sum(-1).mean(1))
            sar.append(((Ds @ P[Dq:, Dq:]) * Ds).sum(-1).mean(1))
            inc.append(2 * ((Ds @ P[Dq:, :Dq]) * Do).sum(-1).mean(1))
    score = np.concatenate(score)
    if P is None:
        return score
    return score, (np.concatenate(opt), np.concatenate(sar), np.concatenate(inc))


def _figura_pca(pca_df, titolo, path):
    x = pca_df["componente"].values
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(x, pca_df["var_spiegata"]); axs[0].set_yscale("log"); axs[0].set_ylabel("varianza spiegata")
    axs[1].bar(x, pca_df["quota_ottico"], width=1, color="tab:green"); axs[1].axhline(.5, color="k", ls="--", lw=.6)
    axs[1].set_ylim(0, 1); axs[1].set_ylabel("quota OTTICO")
    axs[2].bar(x, pca_df["auroc"], width=1, color="tab:red"); axs[2].axhline(.6, color="k", ls="--", lw=.6)
    axs[2].set_ylim(.5, max(.75, pca_df["auroc"].max() + .02)); axs[2].set_ylabel("AUROC componente")
    axs[2].set_xlabel("componente (1 = piu' varianza)")
    for ax in axs:
        ax.set_xscale("log")
    axs[0].set_title(titolo, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)






@handler()
def analisi_spettrale(
    job_name: str = "spettrale_test",
    vectors_from: str = "",                 # job_name separati da virgola (artifact anomaly_results_ocsvm_<job>)
    punti: str = "encoder,pre_proj",
    k: int = 5,                             # vicini per il kNN lw
    tagli: str = "0,1,2,5,10,20,50,100,200,500,1000,2000,4000,8000",   # M = componenti grandi scartate
    max_dim_lw: int = 8192,                 # oltre queste dimensioni (concatenate) la lw viene saltata
    shrink_fallback: float = 0.01,          # shrinkage usato se le dimensioni superano max_dim_lw
    flood_csv_artifact: str = "",           # facoltativo: CSV ID,flood_pct (handler flood_pct_csv)
    min_flood_anom: float = 0.0,            # con il CSV: anomale con % <= di questo escluse
    griglia_pre_proj: int = 4,              
    canali_pre_proj: int = 128, 
):
    k, max_dim_lw = int(k), int(max_dim_lw)
    shrink_fallback, min_flood_anom = float(shrink_fallback), float(min_flood_anom)
    lista_tagli = [int(t) for t in str(tagli).split(",") if t.strip()]
    lista_punti = [p.strip() for p in punti.split(",") if p.strip()]
    project_work = dh.get_project("floods")

    out_root = f"/data/spettrale_{job_name}"
    shutil.rmtree(out_root, ignore_errors=True); os.makedirs(out_root)

    flood = None
    if flood_csv_artifact:
        p = project_work.get_artifact(flood_csv_artifact).download("/data/flood_pct.csv", overwrite=True)
        flood = pd.read_csv(p, dtype={"ID": str}).set_index("ID")["flood_pct"]
        print(f"OK -> CSV percentuali: {len(flood)} serie", flush=True)

    riassunto, curve_minori = {}, {}
    for job in [j.strip() for j in vectors_from.split(",") if j.strip()]:
        src = project_work.get_artifact(f"anomaly_results_ocsvm_{job}").download(f"/data/vec_{job}.zip", overwrite=True)
        with zipfile.ZipFile(src) as z:
            presenti = {pt: z.read(f"{pt}/embeddings.npz") for pt in lista_punti if f"{pt}/embeddings.npz" in z.namelist()}
        os.remove(src)
        for pt in lista_punti:
            if pt not in presenti:
                print(f"manca {pt} in {job}", flush=True)

        for pt, raw in presenti.items():
            nome = f"{job}/{pt}"
            tag = nome.replace("/", "__")
            E = np.load(io.BytesIO(raw))
            if pt == "pre_proj" and griglia_pre_proj > 0:
                # vettori (canali * g * g) -> griglia griglia_pre_proj, media su blocchi: uguale a estrarre con
                # AdaptiveAvgPool2d(griglia_pre_proj) se il pooling e' una media e g e' multiplo della griglia
                E = {key: E[key] for key in E.files}
                g_orig = int(round(np.sqrt(E["fit_q"].shape[1] / canali_pre_proj)))
                if canali_pre_proj * g_orig ** 2 != E["fit_q"].shape[1] or g_orig % griglia_pre_proj:
                    raise ValueError(f"{nome}: {E['fit_q'].shape[1]} dimensioni non riportabili a griglia "
                                     f"{griglia_pre_proj} con {canali_pre_proj} canali")
                if g_orig != griglia_pre_proj:
                    fb = g_orig // griglia_pre_proj
                    for key in ("fit_q", "fit_k", "normal_q", "normal_k", "anom_q", "anom_k"):
                        X_ = E[key]
                        E[key] = X_.reshape(len(X_), canali_pre_proj, griglia_pre_proj, fb, griglia_pre_proj, fb) \
                                   .mean(axis=(3, 5)).reshape(len(X_), -1)
                    print(f"{nome}: griglia {g_orig}x{g_orig} -> {griglia_pre_proj}x{griglia_pre_proj}", flush=True)
            Dq = E["fit_q"].shape[1]
            cat = lambda g: np.concatenate([E[f"{g}_q"], E[f"{g}_k"]], axis=1).astype(np.float32)
            Xf, Xn, Xa = cat("fit"), cat("normal"), cat("anom")
            if flood is not None:
                f = flood.reindex([str(i) for i in E["anom_ids"]]).values
                ok = np.isfinite(f) & (f > min_flood_anom)
                print(f"{nome}: anomale {len(Xa)} -> {ok.sum()} dopo il filtro (% > {min_flood_anom})", flush=True)
                Xa = Xa[ok]
            del E, raw
            D = Xf.shape[1]
            print(f"===== {nome} | fit {len(Xf)} | normali {len(Xn)} | anomale {len(Xa)} | dim {Dq}+{D - Dq} =====", flush=True)
            riga = {"serie fit": len(Xf), "normali": len(Xn), "anomale": len(Xa), "dim OPT": Dq, "dim SAR": D - Dq}

            # ---------------- spettri: concatenato e singole modalita'
            sp = {}
            for blocco, sl in (("OPT+SAR", slice(None)), ("OPT", slice(0, Dq)), ("SAR", slice(Dq, None))):
                sp[blocco] = _spettro(Xf[:, sl], max_dim_lw, shrink_fallback)
                s = sp[blocco]
                print(f"  SVD {blocco}: {len(s['lam'])} componenti"
                      + ("" if s["completo"] else f" (su {Xf[:, sl].shape[1]} dimensioni: piu' dimensioni che serie)")
                      + f" | shrinkage {s['alpha']:.4f}", flush=True)
                riga[f"componenti {blocco}"] = len(s["lam"])
                riga[f"shrinkage {blocco}"] = s["alpha"]

            # ---------------- 1) PCA del concatenato
            s = sp["OPT+SAR"]
            Pn, Pa = _coordinate(s, Xn), _coordinate(s, Xa)
            auroc_pc = _auroc_colonne(Pn, Pa)
            del Pn, Pa
            var = s["lam"] / s["lam"].sum()
            quota = (s["Vt"][:, :Dq].astype(np.float64) ** 2).sum(1)
            pca_df = pd.DataFrame({"componente": np.arange(1, len(var) + 1), "var_spiegata": var,
                                   "quota_ottico": quota, "auroc": auroc_pc})
            pca_df.to_csv(f"{out_root}/pca_{tag}.csv", index=False)
            _figura_pca(pca_df, nome, f"{out_root}/pca_{tag}.png")
            n = len(var); dieci = max(1, n // 10)
            top = np.argsort(auroc_pc)[::-1][:10]
            riga.update({
                "PCA AUROC max componente": auroc_pc.max(),
                "PCA posizione componente migliore": int(np.argmax(auroc_pc) + 1),
                "PCA posizione mediana top-10": float(np.median(top + 1)),
                "PCA AUROC medio primo 10%": auroc_pc[:dieci].mean(),
                "PCA AUROC medio ultimo 10%": auroc_pc[-dieci:].mean(),
                "PCA quota OTTICO primo 10%": quota[:dieci].mean(),
                "PCA quota OTTICO top-10": quota[top].mean(),
            })

            # ---------------- 2) componenti minori
            curve_minori[nome] = {}
            for blocco, sl in (("OPT", slice(0, Dq)), ("SAR", slice(Dq, None)), ("OPT+SAR", slice(None))):
                c = _curva_minori(sp[blocco], Xn[:, sl], Xa[:, sl], lista_tagli)
                curve_minori[nome][blocco] = c
                Mb = max(c, key=c.get)
                riga.update({f"minori {blocco} AUROC M=0": c[0], f"minori {blocco} M migliore": Mb,
                             f"minori {blocco} AUROC M migliore": c[Mb]})
                print(f"  minori {blocco}: M=0 {c[0]:.3f} | migliore M={Mb} {c[Mb]:.3f}", flush=True)
            pd.DataFrame(curve_minori[nome]).rename_axis("M").to_csv(f"{out_root}/minori_{tag}.csv")

            # ---------------- 3) kNN lw
            if D > max_dim_lw or not sp["OPT+SAR"]["completo"]:
                motivo = f"{D} > max_dim_lw" if D > max_dim_lw else "piu' dimensioni che serie di fit"
                print(f"  lw saltata ({motivo})", flush=True)
            else:
                W, P = _lw_da_spettro(sp["OPT+SAR"])
                sn, (on, sn_, inc_n) = _knn_lw(Xn, Xf, sp["OPT+SAR"], W, P, Dq, k)
                sa, (oa, sa_, inc_a) = _knn_lw(Xa, Xf, sp["OPT+SAR"], W, P, Dq, k)
                del P
                sing, Y = {}, {}
                for blocco, sl in (("OPT", slice(0, Dq)), ("SAR", slice(Dq, None))):
                    Wb, _ = _lw_da_spettro(sp[blocco])
                    sing[blocco] = _auroc(_knn_lw(Xn[:, sl], Xf[:, sl], sp[blocco], Wb, k=k),
                                          _knn_lw(Xa[:, sl], Xf[:, sl], sp[blocco], Wb, k=k))
                    trasf = lambda X, s=sp[blocco], Wb=Wb, sl=sl: ((((X[:, sl] - s["mu"]) / s["sd"]) - s["c"]) @ Wb).astype(np.float32)
                    Y[blocco] = trasf
                bl = lambda X: np.concatenate([Y["OPT"](X), Y["SAR"](X)], axis=1)
                ident = {"mu": 0.0, "sd": 1.0, "c": 0.0}
                Bf = bl(Xf); I = np.eye(Bf.shape[1])
                sing["a blocchi"] = _auroc(_knn_lw(bl(Xn), Bf, ident, I, k=k), _knn_lw(bl(Xa), Bf, ident, I, k=k))
                del Bf, I
                completa = _auroc(sn, sa)
                riga.update({
                    "lw AUROC completa": completa,
                    "lw AUROC parte OPT": _auroc(on, oa), "lw AUROC parte SAR": _auroc(sn_, sa_),
                    "lw AUROC parte incrociata": _auroc(inc_n, inc_a),
                    "lw AUROC solo OPT": sing["OPT"], "lw AUROC solo SAR": sing["SAR"],
                    "lw AUROC a blocchi": sing["a blocchi"],
                    "lw quota OPT normali": (on / sn).mean(), "lw quota OPT anomale": (oa / sa).mean(),
                    "lw quota SAR normali": (sn_ / sn).mean(), "lw quota SAR anomale": (sa_ / sa).mean(),
                    "lw quota incrociata normali": (inc_n / sn).mean(), "lw quota incrociata anomale": (inc_a / sa).mean(),
                    "lw guadagno vicini congiunti": sing["a blocchi"] - max(sing["OPT"], sing["SAR"]),
                    "lw guadagno correlazione": completa - sing["a blocchi"],
                })
                print(f"  lw: completa {completa:.3f} | a blocchi {sing['a blocchi']:.3f} | "
                      f"solo OPT {sing['OPT']:.3f} | solo SAR {sing['SAR']:.3f}", flush=True)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(on, sn_, s=5, alpha=.4, label="normali")
                ax.scatter(oa, sa_, s=5, alpha=.4, color="tab:red", label="anomale")
                ax.set_xlabel("parte OTTICA (lw)"); ax.set_ylabel("parte SAR (lw)"); ax.set_title(nome, fontsize=10)
                ax.legend(markerscale=3); fig.tight_layout(); fig.savefig(f"{out_root}/lw_scatter_{tag}.png", dpi=110); plt.close(fig)
            riassunto[nome] = riga
            del sp, Xf, Xn, Xa

    if not riassunto:
        raise ValueError("nessun embedding trovato")

    tab = pd.DataFrame(riassunto)
    tab.to_csv(f"{out_root}/tabella.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_rows", 200):
        print("\n===== TABELLA =====\n", tab.round(3).to_string(), flush=True)

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
    for ax, blocco in zip(axs, ("OPT", "SAR", "OPT+SAR")):
        for nome, cur in curve_minori.items():
            c = cur[blocco]; ax.plot(list(c), list(c.values()), marker="o", label=nome)
        ax.set_xscale("symlog", linthresh=1); ax.grid(alpha=.3)
        ax.set_xlabel("M = componenti piu' grandi scartate"); ax.set_title(blocco)
    axs[0].set_ylabel("AUROC"); axs[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out_root}/minori_confronto.png", dpi=110); plt.close(fig)

    with open(f"{out_root}/parametri.json", "w") as f:
        json.dump({"vectors_from": vectors_from, "punti": punti, "k": k, "tagli": lista_tagli,
                   "max_dim_lw": max_dim_lw, "flood_csv_artifact": flood_csv_artifact,
                   "min_flood_anom": min_flood_anom, "griglia_pre_proj": griglia_pre_proj,
                   "canali_pre_proj": canali_pre_proj}, f, indent=2)
    shutil.make_archive(out_root, "zip", out_root)
    project_work.log_artifact(name=f"spettrale_{job_name}", source=f"{out_root}.zip", kind="artifact")
    shutil.rmtree(out_root, ignore_errors=True); os.remove(f"{out_root}.zip")
    return f"TERMINATO -> spettrale_{job_name}"