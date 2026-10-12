import contextlib
import io
import json
import os
import shutil
import sys
import zipfile
import digitalhub as dh
from digitalhub_runtime_python import handler
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# Funzioni con prefisso _dg_: src e' un unico namespace condiviso tra gli handler, il prefisso evita collisioni.


# ----------------------------------------------------------------------------------------------
# metriche
# ----------------------------------------------------------------------------------------------
def _dg_auroc_low(s_normal, s_anom):
    """AUROC con 'punteggio basso = anomalia': P(punteggio normale > punteggio anomala), 0.5 = caso."""
    s_normal, s_anom = np.asarray(s_normal, dtype=np.float64), np.asarray(s_anom, dtype=np.float64)
    n1, n2 = len(s_normal), len(s_anom)
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = rankdata(np.concatenate([s_normal, s_anom]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def _dg_auroc_se(a, n1, n2):
    """Errore standard dell'AUROC (Hanley-McNeil, 1982)."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float(np.sqrt(max(a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a), 0.0) / (n1 * n2)))


def _dg_fmt_auc(s_normal, s_anom):
    a = _dg_auroc_low(s_normal, s_anom)
    if np.isnan(a):
        return "n/d"
    return f"{a:.3f}±{_dg_auroc_se(a, len(s_normal), len(s_anom)):.3f}"


def _dg_pc(x):
    """Percentuale formattata, 'n/d' se non calcolabile."""
    return "n/d" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


# ----------------------------------------------------------------------------------------------
# statistiche per serie
# ----------------------------------------------------------------------------------------------
def _dg_center_crop(arr, patch_size):
    """Ritaglio centrale sugli ultimi due assi, come fa il loader (nessun taglio se l'immagine e' piu' piccola)."""
    h, w = arr.shape[-2:]
    sy, sx = max(0, (h - patch_size) // 2), max(0, (w - patch_size) // 2)
    return arr[..., sy:sy + patch_size, sx:sx + patch_size]


def _dg_array_stats(arr, prefix, patch_size, black_thr):
    """Statistiche di una serie (canali, istanti, H, W) nella zona che vede il modello (ritaglio centrale).
    zero_t = quota di pixel esattamente 0 per ogni istante (tutti i canali insieme); un istante e' 'nero' se zero_t >= black_thr."""
    if arr.ndim != 4:
        raise ValueError(f"array con {arr.ndim} dimensioni invece di 4 (canali, istanti, H, W)")
    out = {f"{prefix}_H": int(arr.shape[-2]), f"{prefix}_W": int(arr.shape[-1]), f"{prefix}_T": int(arr.shape[1])}
    a = _dg_center_crop(arr, patch_size)
    zero_t = (a == 0).mean(axis=(0, 2, 3))
    for t, z in enumerate(zero_t):
        out[f"{prefix}_zero_t{t}"] = float(z)
    out[f"{prefix}_n_black"] = int((zero_t >= black_thr).sum())
    out[f"{prefix}_zero_mean"] = float(zero_t.mean())
    out[f"{prefix}_zero_max"] = float(zero_t.max())
    out[f"{prefix}_mean"] = float(a.mean(dtype=np.float64))
    out[f"{prefix}_std"] = float(a.std(dtype=np.float64))
    out[f"{prefix}_sat"] = float((a >= 1).mean())
    out[f"{prefix}_nan"] = float(np.isnan(a).mean())
    return out


def _dg_list_ids(root):
    """ID presenti sia in OPT/ sia in SAR/ (partNN.zip di .npy)."""
    per = {}
    for modality in ("OPT", "SAR"):
        ids = set()
        for part in sorted(glob(os.path.join(root, modality, "part*.zip"))):
            with zipfile.ZipFile(part) as z:
                ids.update(n[:-4] for n in z.namelist() if n.endswith(".npy"))
        per[modality] = ids
    return per["OPT"] & per["SAR"]


def _dg_collect(root, ids, patch_size, black_thr, log_every=500):
    """Statistiche di ogni ID leggendo direttamente dagli zip root/{OPT,SAR}/partNN.zip (file {ID}.npy), senza estrarre."""
    wanted = {f"{i}.npy": i for i in ids}
    rows, failed = {}, []
    for modality, prefix in (("OPT", "opt"), ("SAR", "sar")):
        n = 0
        for part in sorted(glob(os.path.join(root, modality, "part*.zip"))):
            with zipfile.ZipFile(part) as z:
                for name in z.namelist():
                    ID = wanted.get(name)
                    if ID is None:
                        continue
                    try:
                        arr = np.load(io.BytesIO(z.read(name)))
                        rows.setdefault(ID, {}).update(_dg_array_stats(arr, prefix, patch_size, black_thr))
                    except Exception as e:
                        failed.append((ID, modality, str(e)[:120]))
                    n += 1
                    if n % log_every == 0:
                        print(f"  {modality}: {n}/{len(wanted)} serie lette", flush=True)
        print(f"OK -> {modality}: {n}/{len(wanted)} serie lette in {os.path.basename(os.path.normpath(root))}", flush=True)
    return rows, failed


def _dg_flood_px_from_masks(mask_dir, flood_class=2):
    """Pixel di classe `flood_class` per ogni maschera in MASK/ (partNN.zip di .npy)."""
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
    return pd.DataFrame(rows, columns=["ID", "flood_px", "total_px"])


# ----------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------
def _dg_black_section(df, black_thr, summary):
    """Tabella degli istanti neri per normali e anomale + dimensioni degli array; aggiorna `summary`."""
    nor, ano = (df.label == 0).values, (df.label == 1).values
    black_opt, black_sar = (df.opt_n_black >= 1).values, (df.sar_n_black >= 1).values
    print(f"=== Istanti neri: un istante e' 'nero' se almeno il {100 * black_thr:.0f}% dei suoi pixel vale esattamente 0 ===")
    rows = []
    for name, g in (("normali", nor), ("anomale", ano)):
        rows.append({"gruppo": name, "n": int(g.sum()),
                     "serie con >=1 istante OTTICO nero": _dg_pc(float(black_opt[g].mean())),
                     "serie con >=1 istante SAR nero": _dg_pc(float(black_sar[g].mean())),
                     "ottico: zeri nell'istante peggiore (mediana)": _dg_pc(float(np.median(df.opt_zero_max.values[g]))),
                     "ottico: zeri nell'istante peggiore (p90)": _dg_pc(float(np.percentile(df.opt_zero_max.values[g], 90))),
                     "SAR: zeri nell'istante peggiore (mediana)": _dg_pc(float(np.median(df.sar_zero_max.values[g])))})
    print(pd.DataFrame(rows).to_string(index=False))
    summary["opt_nero_normali"] = float(black_opt[nor].mean())
    summary["opt_nero_anomale"] = float(black_opt[ano].mean())
    summary["sar_nero_normali"] = float(black_sar[nor].mean())
    summary["sar_nero_anomale"] = float(black_sar[ano].mean())
    for name, g in (("normali", nor), ("anomale", ano)):
        sizes = df[g].groupby(["opt_H", "opt_W"]).size().sort_values(ascending=False).head(4)
        print(f"Dimensioni array ottico (H x W) {name}: " + ", ".join(f"{int(h)}x{int(w)}: {n}" for (h, w), n in sizes.items()))
    print()



def _dg_report(df, job, thr, thr_desc, black_thr, flood_df=None, n_flood_bins=3, min_group=30):
    """Stampa il report e restituisce un dizionario riassuntivo. df: ID, label (0 normale, 1 anomala), sim (bassa = anomalia)
    e le colonne opt_* / sar_* di _dg_array_stats."""
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 70)
    nor, ano = (df.label == 0).values, (df.label == 1).values
    sim = df.sim.values.astype(np.float64)
    n1, n2 = int(nor.sum()), int(ano.sum())
    black_opt, black_sar = (df.opt_n_black >= 1).values, (df.sar_n_black >= 1).values
    black_any = black_opt | black_sar
    summary = {"n_normali": n1, "n_anomale": n2}

    # 1) punto di partenza
    a = _dg_auroc_low(sim[nor], sim[ano])
    summary["auroc"] = a
    ref = job.get("auroc_cos_sim_low_anom", job.get("auroc_ocsvm_low_anom"))
    print(f"=== Serie lette: {n1} normali, {n2} anomale ===")
    print(f"AUROC della similarita' (bassa = anomalia) sulle serie lette: {a:.3f}"
          + (f" | nel JSON del job: {ref:.3f}" if ref is not None else ""))
    if thr is not None:
        flag = sim < thr
        tpr, fpr = float(flag[ano].mean()), float(flag[nor].mean())
        summary.update({"tpr": tpr, "fpr": fpr})
        print(f"Regola del job: {thr_desc} -> anomale prese (TPR) {_dg_pc(tpr)}, falsi allarmi sulle normali (FPR) {_dg_pc(fpr)}")
    else:
        flag = None
        print("Soglia del job non disponibile nel JSON: salto le tabelle sulla regola.")
    print()

    # 2) quante immagini nere ci sono, e di che dimensione sono gli array
    _dg_black_section(df, black_thr, summary)

    # 3) quanto della regola e' dovuto alle serie con istante ottico nero
    if flag is not None:
        print("=== Chi viene segnalato dalla regola: serie con e senza istante ottico nero ===")
        rows = []
        for name, g in (("normali senza istante ottico nero", nor & ~black_opt), ("normali con istante ottico nero", nor & black_opt),
                        ("anomale senza istante ottico nero", ano & ~black_opt), ("anomale con istante ottico nero", ano & black_opt)):
            rows.append({"gruppo": name, "n": int(g.sum()),
                         "segnalate": _dg_pc(float(flag[g].mean())) if g.any() else "n/d",
                         "similarita' mediana": round(float(np.median(sim[g])), 4) if g.any() else "n/d"})
        print(pd.DataFrame(rows).to_string(index=False))
        p = float(black_opt[ano].mean())
        gb, gn = ano & black_opt, ano & ~black_opt
        tpr_b = float(flag[gb].mean()) if gb.any() else float("nan")
        tpr_n = float(flag[gn].mean()) if gn.any() else float("nan")
        summary.update({"tpr_anomale_con_nero": tpr_b, "tpr_anomale_senza_nero": tpr_n})
        print(f"\nAnomale prese = {_dg_pc(tpr)} = [{_dg_pc(p)} con nero x {_dg_pc(tpr_b)} segnalate] "
              f"+ [{_dg_pc(1 - p)} senza nero x {_dg_pc(tpr_n)} segnalate]")
        if np.isfinite(tpr_b):
            print(f"  -> contributo delle serie con nero: {_dg_pc(p * tpr_b)} su {_dg_pc(tpr)}")
        if fpr < 1:
            need = (tpr - fpr) / (1 - fpr)
            summary["quota_nero_necessaria"] = need
            print(f"Se le immagini nere fossero l'UNICA causa (sempre segnalate, le altre anomale come le normali) servirebbe una quota di "
                  f"anomale con nero pari a {_dg_pc(need)}; osservata: {_dg_pc(p)}.")
        print()

    # 4) AUROC togliendo le serie con immagini nere
    print("=== AUROC della similarita' togliendo le serie con istanti neri (stesse esclusioni per normali e anomale) ===")
    rows = []
    for name, keep in (("tutte le serie", np.ones(len(df), bool)), ("senza istanti ottici neri", ~black_opt),
                       ("senza istanti SAR neri", ~black_sar), ("senza istanti neri (ottico o SAR)", ~black_any)):
        rows.append({"serie considerate": name, "n normali": int((nor & keep).sum()), "n anomale": int((ano & keep).sum()),
                     "AUROC": _dg_fmt_auc(sim[nor & keep], sim[ano & keep])})
        summary[f"auroc [{name}]"] = _dg_auroc_low(sim[nor & keep], sim[ano & keep])
    rows.append({"serie considerate": "solo anomale CON nero ottico vs normali senza", "n normali": int((nor & ~black_opt).sum()),
                 "n anomale": int((ano & black_opt).sum()), "AUROC": _dg_fmt_auc(sim[nor & ~black_opt], sim[ano & black_opt])})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    # 5) relazione graduale con la quota di zeri nell'istante ottico peggiore
    print("=== Similarita' in funzione della quota di zeri nell'istante ottico peggiore ===")
    zmax = df.opt_zero_max.values.astype(np.float64)
    edges = [-1e-9, 0.05, 0.2, 0.5, black_thr, 1.0 + 1e-9]
    labels = ["<=5%", "5-20%", "20-50%", f"50-{100 * black_thr:.0f}%", f">={100 * black_thr:.0f}% (nero)"]
    bins = pd.cut(zmax, bins=edges, labels=labels, include_lowest=True)
    rows = []
    for lab in labels:
        g = (bins == lab)
        row = {"zeri nell'istante peggiore": lab, "n normali": int((g & nor).sum()), "n anomale": int((g & ano).sum()),
               "sim. mediana normali": round(float(np.median(sim[g & nor])), 4) if (g & nor).any() else "n/d",
               "sim. mediana anomale": round(float(np.median(sim[g & ano])), 4) if (g & ano).any() else "n/d"}
        if flag is not None:
            row["segnalate normali"] = _dg_pc(float(flag[g & nor].mean())) if (g & nor).any() else "n/d"
            row["segnalate anomale"] = _dg_pc(float(flag[g & ano].mean())) if (g & ano).any() else "n/d"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    for name, g in (("normali", nor), ("anomale", ano)):
        if np.ptp(zmax[g]) > 0 and np.ptp(sim[g]) > 0:
            rho, _ = spearmanr(sim[g], zmax[g])
            summary[f"rho(sim, zeri ottico) {name}"] = float(rho)
            print(f"Correlazione di Spearman (similarita', zeri nell'istante ottico peggiore), {name}: {rho:.3f}")
    print()

    # 6) frazione di flood (solo se sono state fornite le maschere)
    flood_table = None
    if flood_df is not None:
        print("=== AUROC per frazione di flood delle anomale (contro tutte le normali) ===")
        f = flood_df.drop_duplicates("ID").set_index("ID")
        total = f["total_px"] if "total_px" in f.columns else pd.Series(65536, index=f.index)
        ids_a = df.ID.values[ano]
        cov = np.array([i in f.index for i in ids_a])
        summary["anomale_con_maschera"] = int(cov.sum())
        print(f"Anomale con maschera nell'artifact delle maschere: {int(cov.sum())} su {n2} "
              f"(le altre non vengono contate: sono assenti dall'artifact, non e' detto che non abbiano flood)")
        if cov.sum() >= min_group:
            frac = np.array([f.loc[i, "flood_px"] / total.loc[i] for i in ids_a[cov]])
            sa, ba = sim[ano][cov], black_opt[ano][cov]
            zero = frac == 0
            pos = frac[~zero]
            groups = []
            if zero.sum() >= min_group:
                groups.append(("senza flood", zero))
            if len(pos) >= min_group:
                edges = np.unique(np.quantile(pos, np.linspace(0, 1, n_flood_bins + 1)))
                for lo, hi in zip(edges[:-1], edges[1:]):
                    g = (~zero) & (frac >= lo) & ((frac <= hi) if hi == edges[-1] else (frac < hi))
                    if g.sum() >= min_group:
                        groups.append((f"{100 * lo:.1f}-{100 * hi:.1f}%", g))
            rows = [{"frazione di flood": name, "n anomale": int(g.sum()), "AUROC": _dg_fmt_auc(sim[nor], sa[g]),
                     "con istante ottico nero": _dg_pc(float(ba[g].mean())), "similarita' mediana": round(float(np.median(sa[g])), 4)}
                    for name, g in groups]
            flood_table = pd.DataFrame(rows)
            print(flood_table.to_string(index=False))
            if np.ptp(frac) > 0 and np.ptp(sa) > 0:
                rho, p_ = spearmanr(sa, frac)
                summary["rho(sim, flood)"] = float(rho)
                print(f"Spearman tra similarita' e frazione di flood (anomale con maschera): {rho:.3f} (se il segnale e' flood: negativo)")
            print("Se l'AUROC e' piatto tra le fasce (compresa 'senza flood'), la similarita' non stava misurando il flood.")
        else:
            print("Troppo poche anomale con maschera per una tabella per fasce.")
        print()
    return summary, flood_table


def _dg_plot(df, thr, black_thr, path, flood_table=None):
    import matplotlib.pyplot as plt
    nor, ano = (df.label == 0).values, (df.label == 1).values
    black_opt = (df.opt_n_black >= 1).values
    ncol = 3 if flood_table is not None and len(flood_table) else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 4.5), squeeze=False)

    ax = axes[0, 0]
    ax.scatter(df.opt_zero_max[nor], df.sim[nor], s=5, alpha=0.35, label="normali", color="#4C78A8")
    ax.scatter(df.opt_zero_max[ano], df.sim[ano], s=5, alpha=0.35, label="anomale", color="#E45756")
    if thr is not None:
        ax.axhline(thr, color="k", lw=1, ls="--", label="soglia del job")
    ax.axvline(black_thr, color="gray", lw=1, ls=":")
    ax.set_xlabel("quota di pixel a zero nell'istante ottico peggiore")
    ax.set_ylabel("similarita'")
    ax.set_title("Similarita' vs immagini ottiche nere")
    ax.legend(fontsize=8, markerscale=3)

    ax = axes[0, 1]
    allv = df.sim.values
    bins = np.linspace(np.nanmin(allv), np.nanmax(allv), 60)
    for name, g, c in (("normali", nor, "#4C78A8"), ("anomale senza nero ottico", ano & ~black_opt, "#E45756"),
                       ("anomale con nero ottico", ano & black_opt, "#F58518")):
        if g.any():
            ax.hist(df.sim.values[g], bins=bins, density=True, alpha=0.5, label=f"{name} ({int(g.sum())})", color=c)
    if thr is not None:
        ax.axvline(thr, color="k", lw=1, ls="--")
    ax.set_xlabel("similarita'")
    ax.set_title("Distribuzione della similarita'")
    ax.legend(fontsize=8)

    if ncol == 3:
        ax = axes[0, 2]
        vals = [float(str(v).split("±")[0]) for v in flood_table["AUROC"] if str(v) != "n/d"]
        ax.plot(range(len(vals)), vals, marker="o", color="#54A24B")
        ax.axhline(0.5, color="k", lw=1)
        ax.set_xticks(range(len(flood_table)))
        ax.set_xticklabels([str(x).replace(" ", "\n", 1) for x in flood_table["frazione di flood"]], fontsize=8)
        ax.set_ylabel("AUROC")
        ax.set_xlabel("frazione di flood delle anomale")
        ax.set_title("AUROC per frazione di flood")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _dg_report_scan(df, black_thr, ref_tpr=0.0, ref_fpr=0.0, dataset_anom="", dataset_std=""):
    """Report senza risultati di un job: solo cosa c'e' nelle immagini (istanti neri, dimensioni, quota di zeri)."""
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 70)
    nor, ano = (df.label == 0).values, (df.label == 1).values
    n1, n2 = int(nor.sum()), int(ano.sum())
    summary = {"modalita": "scansione", "n_normali": n1, "n_anomale": n2}
    print(f"=== Scansione senza job: {n1} normali (campione di Floods_{dataset_std}_crop_norm), "
          f"{n2} anomale (da {dataset_anom}) ===\n")
    _dg_black_section(df, black_thr, summary)

    print("=== Quota di zeri nell'istante ottico peggiore, per gruppo ===")
    zmax = df.opt_zero_max.values.astype(np.float64)
    edges = [-1e-9, 0.05, 0.2, 0.5, black_thr, 1.0 + 1e-9]
    labels = ["<=5%", "5-20%", "20-50%", f"50-{100 * black_thr:.0f}%", f">={100 * black_thr:.0f}% (nero)"]
    bins = pd.cut(zmax, bins=edges, labels=labels, include_lowest=True)
    rows = []
    for lab in labels:
        g = (bins == lab)
        rows.append({"zeri nell'istante peggiore": lab,
                     "n normali": int((g & nor).sum()), "% normali": _dg_pc(float((g & nor).sum() / n1)),
                     "n anomale": int((g & ano).sum()), "% anomale": _dg_pc(float((g & ano).sum() / n2))})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    if ref_tpr > 0 and ref_fpr < 1:
        need = (ref_tpr - ref_fpr) / (1 - ref_fpr)
        obs = summary["opt_nero_anomale"]
        summary["quota_nero_necessaria"] = need
        print("=== Confronto con la regola di riferimento ===")
        print(f"Regola di riferimento: anomale prese {_dg_pc(ref_tpr)}, falsi allarmi {_dg_pc(ref_fpr)}.")
        print(f"Se le immagini ottiche nere fossero l'UNICA causa (sempre segnalate, le altre anomale come le normali) servirebbe una quota "
              f"di anomale con istante ottico nero pari a {_dg_pc(need)}; osservata: {_dg_pc(obs)} "
              f"(nelle normali: {_dg_pc(summary['opt_nero_normali'])}).")
        print("Senza la similarita' per serie non si puo' dire quante di quelle nere fossero davvero segnalate: serve il job.\n")
    return summary, None


def _dg_plot_scan(df, black_thr, path):
    import matplotlib.pyplot as plt
    nor, ano = (df.label == 0).values, (df.label == 1).values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 41)
    for name, g, c in (("normali", nor, "#4C78A8"), ("anomale", ano, "#E45756")):
        ax.hist(df.opt_zero_max.values[g], bins=bins, weights=np.full(int(g.sum()), 1.0 / int(g.sum())), alpha=0.55,
                label=f"{name} ({int(g.sum())})", color=c)
    ax.axvline(black_thr, color="gray", lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("quota di pixel a zero nell'istante ottico peggiore")
    ax.set_ylabel("quota di serie (scala log)")
    ax.set_title("Zeri nell'istante ottico peggiore")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    w = 0.35
    for i, (name, c) in enumerate((("normali", "#4C78A8"), ("anomale", "#E45756"))):
        g = nor if name == "normali" else ano
        vals = [100 * float((df.opt_n_black.values[g] >= 1).mean()), 100 * float((df.sar_n_black.values[g] >= 1).mean())]
        ax.bar(np.arange(2) + (i - 0.5) * w, vals, w, label=name, color=c)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["ottico", "SAR"])
    ax.set_ylabel("% di serie con almeno un istante nero")
    ax.set_title("Serie con istanti neri")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


class _DgTee:
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
def diagnose_dataset_series(
    results_job: str = "",                  # job da spiegare: legge anomaly_results_{results_job} (job coseno: similarities.csv; job OC-SVM: scores.csv); "" = scansione senza job
    dataset_anomalies: str = "",            # artifact con le serie anomale ("" = quello scritto nel JSON del job; obbligatorio se results_job e' vuoto)
    dataset_standard: str = "Standard",     # le normali sono in Floods_{dataset_standard}_crop_norm (come negli altri handler)
    dataset_masks: str = "",                # facoltativo (solo con results_job): artifact con MASK/ per l'AUROC per frazione di flood; "" = salta
    flood_class: int = 2,                   # valore della classe flood nella maschera
    patch_size: int = 256,                  # ritaglio centrale che vede il modello
    black_thr: float = 0.99,                # un istante e' "nero" se almeno questa quota dei suoi pixel vale esattamente 0
    max_series: int = 0,                    # 0 = tutte le serie; N > 0 = al massimo N per gruppo (prova veloce)
    n_normal_scan: int = 0,                 # solo scansione: normali da campionare dall'artifact standard (0 = tante quante le anomale)
    scan_seed: int = 0,                     # solo scansione: seme del campionamento
    ref_tpr: float = 0.0,                   # solo scansione, facoltativi: TPR e FPR di una regola di riferimento, per il confronto con la quota di serie nere
    ref_fpr: float = 0.0,
    job_name: str = "diagnosi_dataset",
    save_dir: str = "/data/diagnosi_dataset",
):
    """
    Perche' un job di anomaly detection si comporta cosi'? Per ogni serie (normali e anomale) legge le immagini e misura quante hanno
    istanti ottici/SAR "neri" (quasi tutti i pixel a 0), le dimensioni degli array e la quota di zeri.
    - Con results_job: lega queste misure alla similarita' del job (chi viene segnalato, quanto della regola dipende dalle serie con
      immagini nere, AUROC togliendo quelle serie e, con le maschere, AUROC per frazione di flood).
    - Senza results_job (scansione): legge TUTTE le serie dell'artifact delle anomale e un campione di normali, e riporta solo cosa
      contengono le immagini; con ref_tpr/ref_fpr confronta la quota di serie nere con quella che servirebbe a spiegare la regola.
    Solo lettura dei dati: nessun modello, nessuna GPU. Risultato: artifact diagnosi_dataset_{job_name}
    (report.txt, stats_series.csv, diagnosi_dataset.png, riassunto.json[, flood_px.csv]).
    """
    import random
    import matplotlib
    matplotlib.use("Agg")

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if not (0.5 < black_thr <= 1.0):
        raise ValueError(f"black_thr deve essere in (0.5, 1], ricevuto {black_thr}")
    scan = not results_job
    if scan and not dataset_anomalies:
        raise RuntimeError("senza results_job (scansione) serve dataset_anomalies: il nome dell'artifact con le serie anomale")
    if scan and dataset_masks:
        print("ATTENZIONE -> dataset_masks ignorato in modalita' scansione (serve results_job)", flush=True)
        dataset_masks = ""

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    work_dir = save_dir + "_" + job_name
    shutil.rmtree(work_dir, ignore_errors=True)
    in_dir = os.path.join(work_dir, "in")
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1) risultati del job: similarita' per serie + parametri della regola (non in scansione)
    job, res, thr, thr_desc = {}, None, None, ""
    if not scan:
        print(f"Download: anomaly_results_{results_job}", flush=True)
        zip_in = project_work.get_artifact(f"anomaly_results_{results_job}").download(
            os.path.join(in_dir, "anomaly_results.zip"), overwrite=True)
        res_dir = os.path.join(in_dir, "results")
        with zipfile.ZipFile(zip_in) as z:
            z.extractall(res_dir)
        json_path = os.path.join(res_dir, "anomaly_results.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                job = json.load(f)

        if os.path.exists(os.path.join(res_dir, "similarities.csv")):
            res = pd.read_csv(os.path.join(res_dir, "similarities.csv")).rename(columns={"similarity": "sim"})
            if isinstance(job.get("thr"), dict) and "thr_low" in job["thr"]:
                thr = float(job["thr"]["thr_low"])
                thr_desc = f"similarita' < media - {job['thr'].get('n_std', 1):g}*std = {thr:.4f}"
        elif os.path.exists(os.path.join(res_dir, "scores.csv")):
            res = pd.read_csv(os.path.join(res_dir, "scores.csv")).rename(columns={"score": "sim"})
            thr, thr_desc = 0.0, "score OC-SVM < 0"
        else:
            raise RuntimeError(f"ne' similarities.csv ne' scores.csv in anomaly_results_{results_job} (contenuto: {os.listdir(res_dir)})")
        missing_cols = {"ID", "label", "sim"} - set(res.columns)
        if missing_cols:
            raise RuntimeError(f"colonne mancanti nel CSV del job: {sorted(missing_cols)} (trovate: {list(res.columns)})")
        res = res.dropna(subset=["sim"]).copy()
        res["ID"] = res["ID"].astype(str)
        res["label"] = res["label"].astype(int)
        if max_series > 0:
            res = pd.concat([res[res.label == l].sample(min(max_series, int((res.label == l).sum())), random_state=0) for l in (0, 1)])
        print(f"OK -> job {job.get('job_name', results_job)}: {int((res.label == 0).sum())} normali, {int((res.label == 1).sum())} anomale", flush=True)

    anom_name = dataset_anomalies or job.get("dataset_anomalies", "")
    if not anom_name:
        raise RuntimeError("dataset_anomalies vuoto e assente nel JSON del job: passalo come parametro")

    # 2) lettura delle immagini e statistiche per serie
    stats, failed, picked = {}, [], {}
    order = ((1, anom_name, "anomale"), (0, f"Floods_{dataset_standard}_crop_norm", "normali")) if scan else \
            ((0, f"Floods_{dataset_standard}_crop_norm", "normali"), (1, anom_name, "anomale"))
    for label, art_name, folder in order:
        print(f"Download: {art_name}", flush=True)
        root = project_data.get_artifact(art_name).download(os.path.join(work_dir, folder))
        if scan:
            pool = sorted(_dg_list_ids(root))
            if not pool:
                raise RuntimeError(f"nessuna serie con OPT e SAR in {art_name} (cercati partNN.zip di .npy in {root}/OPT e {root}/SAR)")
            if label == 1:
                n_take = len(pool) if max_series <= 0 else min(max_series, len(pool))
            else:
                n_take = n_normal_scan if n_normal_scan > 0 else len(picked[1])
                if max_series > 0:
                    n_take = min(n_take, max_series)
                n_take = min(n_take, len(pool))
            ids = pool if n_take >= len(pool) else random.Random(scan_seed).sample(pool, n_take)
            picked[label] = ids
            print(f"OK -> {art_name}: {len(pool)} serie complete, ne leggo {len(ids)}", flush=True)
        else:
            ids = res.ID[res.label == label].tolist()
        rows, fl = _dg_collect(root, ids, patch_size, black_thr)
        stats.update(rows)
        failed += fl
        shutil.rmtree(os.path.join(work_dir, folder), ignore_errors=True)       # libera spazio: serve solo la lettura
    if scan:
        res = pd.DataFrame({"ID": picked[0] + picked[1], "label": [0] * len(picked[0]) + [1] * len(picked[1])})
    if failed:
        print(f"ATTENZIONE -> {len(failed)} letture fallite, prime 5: {failed[:5]}", flush=True)

    st = pd.DataFrame.from_dict(stats, orient="index")
    st.index.name = "ID"
    df = res.merge(st, left_on="ID", right_index=True, how="left")
    need = ["opt_zero_max", "sar_zero_max", "opt_n_black", "sar_n_black"]
    ok = df[need].notna().all(axis=1) if all(c in df.columns for c in need) else pd.Series(False, index=df.index)
    n_lost = int((~ok).sum())
    if n_lost:
        print(f"ATTENZIONE -> {n_lost} serie senza immagini OPT+SAR complete negli artifact (escluse dal report): "
              f"{int(((~ok) & (df.label == 0)).sum())} normali, {int(((~ok) & (df.label == 1)).sum())} anomale", flush=True)
    df = df[ok].reset_index(drop=True)
    for l, nm in ((0, "normale"), (1, "anomala")):
        if int((df.label == l).sum()) == 0:
            raise RuntimeError(f"nessuna serie {nm} letta: controlla dataset_standard='{dataset_standard}' e dataset_anomalies='{anom_name}'")
    df.to_csv(os.path.join(out_dir, "stats_series.csv"), index=False)

    # 3) maschere (facoltative)
    flood_df = None
    if dataset_masks:
        print(f"Download: {dataset_masks} (serve solo MASK/)", flush=True)
        mpath = project_data.get_artifact(dataset_masks).download(os.path.join(work_dir, "masks"))
        mask_dir = os.path.join(mpath, "MASK")
        if not os.path.isdir(mask_dir):
            raise RuntimeError(f"cartella MASK/ non trovata in {dataset_masks} (contenuto: {os.listdir(mpath)})")
        flood_df = _dg_flood_px_from_masks(mask_dir, flood_class=flood_class)
        flood_df.to_csv(os.path.join(out_dir, "flood_px.csv"), index=False)
        print(f"OK -> {len(flood_df)} maschere lette da {dataset_masks}/MASK", flush=True)
        shutil.rmtree(os.path.join(work_dir, "masks"), ignore_errors=True)

    # 4) report (stampa sul log e su report.txt)
    with open(os.path.join(out_dir, "report.txt"), "w") as rep:
        with contextlib.redirect_stdout(_DgTee(sys.stdout, rep)):
            if scan:
                summary, flood_table = _dg_report_scan(df, black_thr, ref_tpr, ref_fpr, anom_name, dataset_standard)
            else:
                print(f"Anomale: {anom_name} | normali: Floods_{dataset_standard}_crop_norm")
                summary, flood_table = _dg_report(df, job, thr, thr_desc, black_thr, flood_df=flood_df)
    if scan:
        _dg_plot_scan(df, black_thr, os.path.join(out_dir, "diagnosi_dataset.png"))
    else:
        _dg_plot(df, thr, black_thr, os.path.join(out_dir, "diagnosi_dataset.png"), flood_table)

    summary.update({"results_job": results_job, "dataset_anomalies": anom_name, "dataset_standard": dataset_standard,
                    "serie_escluse_per_dati_mancanti": n_lost, "letture_fallite": len(failed), "black_thr": black_thr})
    with open(os.path.join(out_dir, "riassunto.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))

    zip_base = work_dir.rstrip("/") + "_out"
    shutil.make_archive(zip_base, "zip", out_dir)
    project_work.log_artifact(
        name=f"diagnosi_dataset_{job_name}",
        kind="artifact",
        source=f"{zip_base}.zip",
    )
    print(f"OK -> artifact diagnosi_dataset_{job_name}: {sorted(os.listdir(out_dir))}", flush=True)

    return "TERMINATO"