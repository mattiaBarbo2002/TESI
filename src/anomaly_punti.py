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
from sklearn.covariance import LedoitWolf

from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_2d
from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_attn
from floodlib.loader import MoCo2encodersLoader
from floodlib.builder import MoCo2encoders_2d
from floodlib.builder import MoCo2encoders_attn


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
def ad_md(
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

    whitening: str = "pca",                         # "pca" | "lw"
    fit_on_train: bool = True,                      # True: fit su tutte le serie di train di moco
    max_fit_series: int = 0,                        # 0 = tutte; altrimenti campione casuale (ocsvm lento con molte serie)
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    # punti selezionati e componenti pca per punto
    punti = []

    if do_encoder:
        punti.append("encoder")
    if do_pre_proj:
        punti.append("pre_proj")
    if do_proj_raw:
        punti.append("proj_raw")
    if do_post_proj:
        punti.append("post_proj")

    pca_punto = {"encoder": pca_encoder, "pre_proj": pca_pre_proj, "proj_raw": pca_proj_raw, "post_proj": pca_post_proj}
    
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
    emb = {punto: {} for punto in punti}

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

            emb[punto]["fit"] = take(normal_fit_ids, "fit")
            emb[punto]["thr"] = take(normal_thr_ids, "thr")
            emb[punto]["normal"] = take(normal_test_ids, "normali test")
            emb[punto]["anom"] = take(anom_ids, "anomale")
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
        print("Embedding MoCo: serie di fit pca_md", flush=True)
        emb = embed(normal_fit_ids)
        for punto in punti: emb[punto]["fit"] = emb[punto]
        print_dim("normali fit", emb)

        # embedding serie thr
        print("Embedding MoCo: serie thr pca_md", flush=True)
        emb = embed(normal_thr_ids)
        for punto in punti: emb[punto]["thr"] = emb[punto]
        print_dim("normali thr", emb)

        # embedding serie normali
        print("Embedding MoCo: serie normali", flush=True)
        emb = embed(normal_test_ids)
        for punto in punti: emb[punto]["normal"] = emb[punto]
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
        q_fit, k_fit = emb[punto]["fit"]
        q_thr, k_thr = emb[punto]["thr"]
        q_normal, k_normal = emb[punto]["normal"]
        q_anom, k_anom = emb[punto]["anom"]

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