import contextlib
import io
import json
import os
import shutil
import sys
import zipfile
from digitalhub_runtime_python import handler
 
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
 
 
@handler()
def regress_flood_from_embeddings(
    results_job: str = "net_v3_ocsvm",      # job OC-SVM con embeddings.npz: legge l'artifact anomaly_results_{results_job}
    analysis_job: str = "",                 # facoltativo: nome di un'analisi gia' fatta (artifact analisi_embeddings_{analysis_job}) da cui prendere flood_px.csv, senza riscaricare le maschere
    dataset_masks: str = "",                # altrimenti: artifact con MASK/ ("" = dataset_anomalies scritto nel JSON del job)
    flood_class: int = 2,                   # valore della classe flood nella maschera
    n_splits: int = 5,                      # fold della cross-validation (a gruppi: le serie dello stesso evento restano insieme)
    job_name: str = "regressione_net_v3_ocsvm",
    save_dir: str = "/data/regressione_flood",
):
    """
    Quanta informazione sulla frazione di flood c'e' nei vettori? Usa SOLO le serie anomale (nessuna normale) e prevede la frazione
    di flood di ogni serie da q (ottico), da k (SAR) e da [q;k] con una regressione lineare Ridge (feature standardizzate, penalita'
    scelta da una validazione interna), in cross-validation a gruppi per evento. Sulle previsioni fuori campione riporta:
      rho  = correlazione di Spearman con la frazione vera (l'ordine delle serie per flood e' indovinato?), con intervallo al 95%
             ottenuto ricampionando gli eventi;
      R2   = quanto l'errore al quadrato e' piu' basso di quello di chi prevede sempre la media (puo' essere negativo);
      MAE  = errore assoluto medio in punti percentuali di flood, accanto a quello di chi prevede sempre la mediana.
    Il flood e le maschere servono solo a misurare: nessuna etichetta entra nel rilevatore.
    Risultato: artifact regressione_flood_{job_name} (report.txt, regressione_flood.csv, previsioni_fuori_campione.csv,
    regressione_flood.png, riassunto.json[, flood_px.csv]).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
 
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
 
    # ------------------------------------------------------------------------------------------
    # 1) embeddings delle anomale
    # ------------------------------------------------------------------------------------------
    print(f"Download: anomaly_results_{results_job}", flush=True)
    zip_in = project_work.get_artifact(f"anomaly_results_{results_job}").download(
        os.path.join(in_dir, "anomaly_results.zip"), overwrite=True)
    res_dir = os.path.join(in_dir, "results")
    with zipfile.ZipFile(zip_in) as z:
        z.extractall(res_dir)
    npz_path = os.path.join(res_dir, "embeddings.npz")
    if not os.path.exists(npz_path):
        raise RuntimeError(f"embeddings.npz non trovato in anomaly_results_{results_job} (contenuto: {os.listdir(res_dir)}): "
                           f"serve un job eseguito con la versione di train_anomaly_detection_ocsvm che lo salva")
    job = {}
    json_path = os.path.join(res_dir, "anomaly_results.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            job = json.load(f)
 
    with np.load(npz_path) as d:
        missing = [key for key in ("anom_ids", "anom_q", "anom_k") if key not in d.files]
        if missing:
            raise RuntimeError(f"nell'npz mancano {missing} (presenti: {d.files})")
        ids = [str(i) for i in d["anom_ids"]]
        q = np.asarray(d["anom_q"], dtype=np.float64)
        k = np.asarray(d["anom_k"], dtype=np.float64)
    q, k = q.reshape(len(q), -1), k.reshape(len(k), -1)
    if len(q) != len(ids) or len(k) != len(ids):
        raise ValueError(f"anom_q, anom_k e anom_ids hanno lunghezze diverse ({len(q)}, {len(k)}, {len(ids)})")
    print(f"OK -> {len(ids)} anomale, dim q = {q.shape[1]}, dim k = {k.shape[1]}", flush=True)
 
    # ------------------------------------------------------------------------------------------
    # 2) frazione di flood per serie: da un'analisi precedente (veloce) o dalle maschere (scarica l'artifact)
    # ------------------------------------------------------------------------------------------
    flood_csv = os.path.join(out_dir, "flood_px.csv")
    if analysis_job:
        print(f"Download: analisi_embeddings_{analysis_job} (solo flood_px.csv)", flush=True)
        zip_an = project_work.get_artifact(f"analisi_embeddings_{analysis_job}").download(
            os.path.join(in_dir, "analisi.zip"), overwrite=True)
        an_dir = os.path.join(in_dir, "analisi")
        with zipfile.ZipFile(zip_an) as z:
            z.extractall(an_dir)
        src_csv = os.path.join(an_dir, "flood_px.csv")
        if not os.path.exists(src_csv):
            raise RuntimeError(f"flood_px.csv non trovato in analisi_embeddings_{analysis_job} (contenuto: {os.listdir(an_dir)}): "
                               f"lascia analysis_job vuoto e passa dataset_masks")
        flood_df = pd.read_csv(src_csv)
        flood_df.to_csv(flood_csv, index=False)
    else:
        masks_name = dataset_masks or job.get("dataset_anomalies", "")
        if not masks_name:
            raise RuntimeError("dataset_masks vuoto e dataset_anomalies assente nel JSON del job: passa dataset_masks o analysis_job")
        print(f"Download: {masks_name} (serve solo MASK/)", flush=True)
        dataset_path = project_data.get_artifact(masks_name).download(os.path.join(work_dir, "dataset"))
        mask_dir = os.path.join(dataset_path, "MASK")
        if not os.path.isdir(mask_dir):
            raise RuntimeError(f"cartella MASK/ non trovata in {masks_name} (contenuto: {os.listdir(dataset_path)}): "
                               f"serve un artifact costruito con la nuova crop_norm_artifact_anomalies, passalo in dataset_masks")
        mask_rows = []
        for part_name in sorted(os.listdir(mask_dir)):
            if part_name.startswith("part") and part_name.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(mask_dir, part_name)) as z:
                    for member in z.namelist():
                        if member.endswith(".npy"):
                            m = np.load(io.BytesIO(z.read(member)))
                            mask_rows.append((member[:-4], int((m == flood_class).sum()), int(m.size)))
        if not mask_rows:
            raise RuntimeError(f"nessun partNN.zip con .npy in {mask_dir}")
        flood_df = pd.DataFrame(mask_rows, columns=["ID", "flood_px", "total_px"])
        flood_df.to_csv(flood_csv, index=False)
        print(f"OK -> {len(flood_df)} maschere lette da {masks_name}/MASK", flush=True)
        shutil.rmtree(os.path.join(work_dir, "dataset"), ignore_errors=True)
    n_cov = len(set(ids) & set(flood_df.ID))
    print(f"OK -> frazione di flood disponibile per {n_cov}/{len(ids)} anomale", flush=True)
    if n_cov == 0:
        raise RuntimeError("nessuna serie anomala ha una maschera: le maschere non corrispondono alle serie del job")
 
    # ------------------------------------------------------------------------------------------
    # 3) regressione. Quello che si stampa qui va sia nel log sia in report.txt: si raccoglie in un buffer e si
    #    scrive alla fine (se c'e' un errore a meta' il buffer viene comunque stampato sul log)
    # ------------------------------------------------------------------------------------------
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            print("=== Regressione lineare (Ridge) della frazione di flood dal vettore, solo serie anomale ===")
 
            # frazione di flood vera di ogni anomala (solo quelle con maschera)
            f = flood_df.drop_duplicates("ID").set_index("ID")
            total = f["total_px"] if "total_px" in f.columns else pd.Series(65536, index=f.index)
            keep = np.array([i in f.index for i in ids])
            n_all, n_keep = len(ids), int(keep.sum())
            print(f"Anomale nell'npz: {n_all} | con maschera: {n_keep}")
            if n_keep < 3 * n_splits:
                raise ValueError(f"troppo poche anomale con maschera ({n_keep}) per una cross-validation a {n_splits} fold")
            if n_keep < n_all:
                print(f"ATTENZIONE -> {n_all - n_keep} anomale senza maschera, escluse dalla regressione")
            ids_k = [i for i, kp in zip(ids, keep) if kp]
            y = np.array([f.loc[i, "flood_px"] / total.loc[i] for i in ids_k], dtype=np.float64)
            if np.ptp(y) == 0:
                raise ValueError("frazione di flood identica per tutte le anomale: niente da prevedere")
            Xs = {"q": q[keep], "k": k[keep], "[q;k]": np.hstack([q[keep], k[keep]])}
 
            # fold: il gruppo e' la parte dell'ID prima del primo '_' (di solito l'evento), cosi' le serie dello stesso evento
            # stanno nello stesso fold e il modello e' misurato su eventi mai visti
            groups = np.array([str(i).split("_")[0] for i in ids_k])
            n_groups = len(set(groups))
            if n_groups >= n_splits:
                folds = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(ids_k)), groups=groups))
                cv_desc = f"GroupKFold a {n_splits} fold su {n_groups} gruppi (ID prima del primo '_')"
                if n_groups > 0.9 * len(ids_k):
                    cv_desc += (": ATTENZIONE, quasi ogni serie e' un gruppo, quindi nessuna protezione dalla fuga tra serie "
                                "dello stesso evento e i risultati possono essere ottimistici")
            else:
                folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.zeros(len(ids_k))))
                cv_desc = (f"KFold a {n_splits} fold (solo {n_groups} gruppi nell'ID: non basta per raggruppare, "
                           f"serie dello stesso evento possono finire in fold diversi)")
 
            # eventi (per il ricampionamento dell'intervallo di confidenza)
            uniq = np.unique(groups)
            idx_ev = {g: np.where(groups == g)[0] for g in uniq}
 
            rows, preds = [], {}
            for name, X in Xs.items():
                # Ridge fuori campione e, per confronto, chi prevede sempre la MEDIANA del fold di addestramento
                pred, base, alphas = np.zeros(len(y)), np.zeros(len(y)), []
                for tr, te in folds:
                    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))).fit(X[tr], y[tr])
                    pred[te] = m.predict(X[te])
                    base[te] = np.median(y[tr])
                    alphas.append(float(m[-1].alpha_))
                rho = float(spearmanr(pred, y)[0])
 
                # intervallo al 95% di rho ricampionando gli EVENTI (non le serie) con reinserimento: tiene conto del fatto
                # che le serie dello stesso evento si assomigliano. Le previsioni sono quelle fuori campione, niente riaddestramento
                lo, hi = float("nan"), float("nan")
                if len(uniq) >= 5:
                    rng = np.random.default_rng(0)
                    boot = []
                    for _ in range(1000):
                        ii = np.concatenate([idx_ev[uniq[p]] for p in rng.integers(0, len(uniq), len(uniq))])
                        if np.ptp(y[ii]) == 0 or np.ptp(pred[ii]) == 0:
                            continue
                        boot.append(spearmanr(pred[ii], y[ii])[0])
                    if len(boot) >= 100:
                        lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
 
                rows.append({"ingresso": name, "rho": rho, "IC95 rho (eventi) basso": lo, "IC95 rho (eventi) alto": hi,
                             "R2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                             "MAE (punti %)": float(100 * np.abs(y - np.clip(pred, 0, 1)).mean()),
                             "MAE sempre-mediana (punti %)": float(100 * np.abs(y - base).mean()),
                             "alpha mediano": float(np.median(alphas)), "n": int(len(y)), "dim": int(X.shape[1])})
                preds[name] = pred
            table = pd.DataFrame(rows)
            details = {"cv": cv_desc, "n_gruppi": int(n_groups), "n_anomale_con_maschera": n_keep,
                       "flood_mediano_%": float(100 * np.median(y)), "flood_medio_%": float(100 * y.mean())}
 
            pd.DataFrame({"ID": ids_k, "gruppo": groups, "flood_vero": y, "prev_q": preds["q"], "prev_k": preds["k"],
                          "prev_[q;k]": preds["[q;k]"]}).to_csv(os.path.join(out_dir, "previsioni_fuori_campione.csv"), index=False)
            table.to_csv(os.path.join(out_dir, "regressione_flood.csv"), index=False)
 
            # grafico: frazione prevista (fuori campione) contro frazione vera
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
            for ax, (_, r) in zip(axes[0], table.iterrows()):
                ax.scatter(100 * y, 100 * np.clip(preds[r["ingresso"]], 0, 1), s=6, alpha=0.3, color="#4C78A8")
                ax.plot([0, 100], [0, 100], color="k", lw=1, ls="--", label="previsione perfetta")
                ax.set_xlabel("frazione di flood vera (%)")
                ax.set_ylabel("frazione prevista, fuori campione (%)")
                ax.set_title(f"{r['ingresso']}: rho {r['rho']:.2f} | R2 {r['R2']:.2f} | MAE {r['MAE (punti %)']:.1f} punti")
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "regressione_flood.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)
 
            pd.set_option("display.width", 220, "display.max_columns", 30)
            print(f"Cross-validation: {details['cv']}.")
            print(f"Frazione di flood delle anomale: mediana {details['flood_mediano_%']:.1f}%, media {details['flood_medio_%']:.1f}%.\n")
            print(table.round(3).to_string(index=False), "\n")
            print("rho: correlazione di Spearman tra frazione prevista e vera, fuori campione (0 = il vettore non contiene il flood, "
                  "1 = lo contiene; indicativamente rho 0.6 = circa 7 coppie di serie su 10 messe nell'ordine giusto, rho 0.9 = circa 8.5 su 10).")
            print("IC95 (eventi): intervallo ottenuto ricampionando gli eventi, piu' onesto di un errore standard che suppone serie indipendenti"
                  + (" (NaN: servono almeno 5 eventi distinti)." if table["IC95 rho (eventi) basso"].isna().any() else "."))
            print("R2: 1 - errore quadratico / errore di chi prevede sempre la media; puo' essere negativo (peggio della media).")
            print("MAE: errore assoluto medio in punti percentuali di flood (previsioni limitate a 0-100%), da confrontare con la colonna "
                  "'sempre-mediana': se non e' molto piu' basso, il modello non aiuta.")
            print("alpha: forza della penalita' Ridge scelta (mediana sui fold); molto alta = pochi segnali lineari robusti.")
            print("Attenzione: nuvole e altre caratteristiche della scena correlate col flood possono contribuire; la regressione "
                  "usa le etichette, quindi non e' una misura del rilevatore non supervisionato.\n")
    except Exception:
        print(buf.getvalue(), flush=True)
        raise
    report_text = buf.getvalue()
    print(report_text, flush=True)
    with open(os.path.join(out_dir, "report.txt"), "w") as rep:
        rep.write(report_text)
 
    best = table.sort_values("rho", ascending=False).iloc[0]
    records = [{c: (None if (isinstance(v, float) and np.isnan(v)) else v) for c, v in r.items()}
               for r in table.to_dict(orient="records")]       # NaN -> null: JSON valido
    with open(os.path.join(out_dir, "riassunto.json"), "w") as f:
        json.dump({"results_job": results_job, "analysis_job": analysis_job, **details,
                   "migliore": {"ingresso": best["ingresso"], "rho": float(best["rho"]), "R2": float(best["R2"]),
                                "MAE_punti_%": float(best["MAE (punti %)"])},
                   "risultati": records}, f, indent=2, default=float)
 
    zip_base = work_dir.rstrip("/") + "_out"
    shutil.make_archive(zip_base, "zip", out_dir)
    project_work.log_artifact(
        name=f"regressione_flood_{job_name}",
        kind="artifact",
        source=f"{zip_base}.zip",
    )
    print(f"OK -> artifact regressione_flood_{job_name}: {sorted(os.listdir(out_dir))}", flush=True)
 
    return "TERMINATO"
