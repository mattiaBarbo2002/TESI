import os
import zipfile
import shutil
import torch
import numpy as np
import rasterio
import digitalhub as dh
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import zipfile
import re
import xarray as xr
import zarr
import traceback
import tarfile
import io
import csv
from glob import glob
from digitalhub_runtime_python import handler

from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE
from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_2d

matplotlib.use("Agg")  



@handler()
def visual_series(
    patch_size: int = 256,
    n_images1: int = 4,
    n_channels1: int = 2,
    n_images2: int = 4,
    n_channels2: int = 10,
    output_dim: int = 10,
    mamba: bool = False,
    dataset: str = "Test",
    test_sar: bool = True,
    test_opt: bool = True,
    weights_sar: str = "weights_s1",
    weights_opt: str = "weights_s2",
    n_samples: int = 10,
    job_name: str = "nome",
    ltae: bool = False,
    monodimensional: bool = False,
    sar_channel: int = 0,                 # NUOVO: quale canale SAR mostrare in scala di grigi
    recon_channels: list | None = None,   # ORA: terna di indici canale OPT per l'RGB, default [2,1,0] = B4,B3,B2
    save_dir: str = "/data/debug_recon",
):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
 
    torch.backends.cudnn.benchmark = True
 
    if recon_channels is None:
        recon_channels = [2, 1, 0]
 
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    random_sar_ids = []
    random_opt_ids = []
    cache_sar_dir = "/data/cache_SAR"
    cache_opt_dir = "/data/cache_OPT"
 
    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"{dataset}").download("/data/dataset_floods")
    print("OK -> Download terminato", flush=True)
 
    save_dir = save_dir + "_" + job_name
    os.makedirs(save_dir, exist_ok=True)
 
    # ------------------------------------------------------------------
    # visualizzazione input vs ricostruzione: griglia 2x4, riga 0 = input,
    # riga 1 = ricostruito, una colonna per istante temporale
    # ------------------------------------------------------------------
    def _visualizza_input_ricostruzione(input_arr, recon_arr, nome_serie, nome_output, is_sar):
        """
        input_arr, recon_arr: array (n_channels, 4, H, W), gia' in [0,1] (cache crop_norm)
        is_sar=True -> un solo canale in grigio (sar_channel)
        is_sar=False -> RGB con i tre canali di recon_channels (indici su n_channels2)
        """
        try:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f"Serie: {nome_serie}", fontsize=16)
 
            for step in range(4):
                if is_sar:
                    img_in = input_arr[sar_channel, step]
                    img_rec = recon_arr[sar_channel, step]
                    kwargs = {"cmap": "gray", "vmin": 0, "vmax": 1}
                else:
                    r, g, b = recon_channels
                    img_in = np.dstack([input_arr[r, step], input_arr[g, step], input_arr[b, step]])
                    img_rec = np.dstack([recon_arr[r, step], recon_arr[g, step], recon_arr[b, step]])
                    img_rec = np.clip(img_rec, 0.0, 1.0)
                    kwargs = {}
 
                axes[0, step].imshow(img_in, **kwargs)
                axes[0, step].set_title(f"Input t{step+1}")
                axes[0, step].axis('off')
 
                axes[1, step].imshow(img_rec, **kwargs)
                axes[1, step].set_title(f"Ricostruito t{step+1}")
                axes[1, step].axis('off')
 
            plt.tight_layout()
            plt.savefig(nome_output, bbox_inches='tight', dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"EXC -> PNG {nome_serie}: {e}", flush=True)
 
    # indicizzo gli ID gia' pronti nella cache (senza estrarre), poi campiono e estraggo
    try:
        sar_available, opt_available = set(), set()
 
        if test_sar:
            for part_path in sorted(glob(os.path.join(dataset_path, "SAR", "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    sar_available.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
            print(f"{len(sar_available)} serie SAR trovate", flush=True)
 
        if test_opt:
            for part_path in sorted(glob(os.path.join(dataset_path, "OPT", "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    opt_available.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
            print(f"{len(opt_available)} serie OPT trovate", flush=True)
 
        # NUOVO: se testo entrambe le modalita', campiono UNA SOLA VOLTA dalle
        # serie presenti in entrambe, cosi' SAR e OPT mostrano le stesse ID -
        # altrimenti (solo una modalita' richiesta) campiono dal suo solo pool
        if test_sar and test_opt:
            common_available = sar_available & opt_available
            sampled = random.sample(sorted(common_available), min(n_samples, len(common_available))) if common_available else []
            print(f"{len(common_available)} serie presenti in entrambe SAR e OPT, campionate {len(sampled)}", flush=True)
            random_sar_ids = sampled
            random_opt_ids = sampled
        elif test_sar:
            random_sar_ids = random.sample(sorted(sar_available), min(n_samples, len(sar_available))) if sar_available else []
        elif test_opt:
            random_opt_ids = random.sample(sorted(opt_available), min(n_samples, len(opt_available))) if opt_available else []
 
        if test_sar:
            os.makedirs(cache_sar_dir, exist_ok=True)
            wanted_sar = {f"{ID}.npy" for ID in random_sar_ids}
            for part_path in sorted(glob(os.path.join(dataset_path, "SAR", "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_sar:
                            z.extract(name, cache_sar_dir)
 
        if test_opt:
            os.makedirs(cache_opt_dir, exist_ok=True)
            wanted_opt = {f"{ID}.npy" for ID in random_opt_ids}
            for part_path in sorted(glob(os.path.join(dataset_path, "OPT", "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_opt:
                            z.extract(name, cache_opt_dir)
 
    except Exception as e:
        print(f"EXC -> recupero liste: {e}", flush=True)
 
    if test_sar:
        try:
            sar_path = project_work.get_artifact(weights_sar).download("/data/weights_sar.pth")
        except Exception as e:
            print(f"EXC -> {weights_sar} non trovato: {e}", flush=True)
            sar_path = None
 
        if sar_path:
            if monodimensional:
                modelSAR = Singlemodal_CAE(input_dim=n_channels1, output_dim=output_dim, n_images=n_images1, mamba=mamba).to(device)
 
            else:
                modelSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=output_dim, n_images=n_images1, n_head=8, d_k=8, ltae=ltae).to(device)
 
            state_dict = torch.load(sar_path, map_location=device)
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            modelSAR.load_state_dict(state_dict)
            modelSAR.eval()
 
            print("SERIE SAR:", flush=True)
            with torch.no_grad():
                for i, ID in enumerate(random_sar_ids):
                    im = np.load(os.path.join(cache_sar_dir, f"{ID}.npy"))
                    im = torch.from_numpy(im).unsqueeze(0).to(device)   # aggiunge la dimensione batch (gia' croppato/normalizzato)
 
                    if monodimensional:
                        try:
                            latent_vector = modelSAR.encoder(im)
                        except AttributeError as e:
                            print(f"EXC -> nome encoder sar: {e}", flush=True)
                            break
 
                        vector = latent_vector.cpu().numpy().flatten()
                        print(f"ID {i}: {ID} | shape: {list(latent_vector.shape)}", flush=True)
                        print(f"VEC: {np.round(vector, 4)}\n", flush=True)
 
                    # NUOVO: ricostruzione intera (encoder+decoder) e confronto input/output su PNG
                    recon = modelSAR(im)
                    _visualizza_input_ricostruzione(
                        input_arr=im.squeeze(0).cpu().numpy(),
                        recon_arr=recon.squeeze(0).detach().cpu().numpy(),
                        nome_serie=ID,
                        nome_output=os.path.join(save_dir, f"SAR_{ID}_recon.png"),
                        is_sar=True,
                    )

                    print("shape sar input:", im.shape)
                    print("shape sar ricostruzione:", recon.shape)
 
            del modelSAR
            torch.cuda.empty_cache()
 
    if test_opt:
        try:
            opt_path = project_work.get_artifact(weights_opt).download("/data/weights_opt.pth")
        except Exception as e:
            print(f"EXC -> {weights_opt} non trovato: {e}", flush=True)
            opt_path = None
 
        if opt_path:
 
            if monodimensional:
                modelOPT = Singlemodal_CAE(input_dim=n_channels2, output_dim=output_dim, n_images=n_images2, mamba=mamba).to(device)
 
            else:
                modelOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=output_dim, n_images=n_images2, n_head=8, d_k=8, ltae=ltae).to(device)
 
            state_dict = torch.load(opt_path, map_location=device)
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            modelOPT.load_state_dict(state_dict)
            modelOPT.eval()
 
            print("SERIE OPT:", flush=True)
            with torch.no_grad():
                for i, ID in enumerate(random_opt_ids):
                    im = np.load(os.path.join(cache_opt_dir, f"{ID}.npy"))
                    im = torch.from_numpy(im).unsqueeze(0).to(device)
 
                    if monodimensional:
                        try:
                            latent_vector = modelOPT.encoder(im)
                        except AttributeError as e:
                            print(f"EXC -> nome encoder opt: {e}", flush=True)
                            break
 
                        vector = latent_vector.cpu().numpy().flatten()
                        print(f"ID {i}: {ID} | shape: {list(latent_vector.shape)}", flush=True)
                        print(f"VEC: {np.round(vector, 4)}\n", flush=True)
 
                    # NUOVO: ricostruzione intera e confronto input/output su PNG
                    recon = modelOPT(im)
                    _visualizza_input_ricostruzione(
                        input_arr=im.squeeze(0).cpu().numpy(),
                        recon_arr=recon.squeeze(0).detach().cpu().numpy(),
                        nome_serie=ID,
                        nome_output=os.path.join(save_dir, f"OPT_{ID}_recon.png"),
                        is_sar=False,
                    )

                    print("shape opt input:", im.shape)
                    print("shape opt ricostruzione:", recon.shape)
 
            del modelOPT
            torch.cuda.empty_cache()
 
    # zip png, upload artifact
    zip_base = save_dir.rstrip("/")
    zip_path = f"{zip_base}.zip"
    try:
        shutil.make_archive(zip_base, 'zip', save_dir)
        project_work.log_artifact(
            name=f"PNG_{job_name}_{dataset}",
            kind="artifact",
            source=zip_path
        )
        print(f"OK -> {zip_path} caricato come artifact", flush=True)
    except Exception as e:
        print(f"EXC -> upload zip ricostruzioni: {e}", flush=True)
 
    return "TERMINATO"


@handler()
def visualize_mask(
    dataset: str = "Floods_Anomalies_v2_crop_norm_full",   # artifact con SAR/, OPT/, OPT_RGB/, MASK/ (partNN.zip di .npy)
    n_samples: int = 10,
    min_flood_pixels: int = 1000,    # considera solo serie con almeno questi pixel classe flood (0 = nessun filtro)
    flood_class: int = 2,            # valore della classe flood nella maschera
    flood_instant: int = 3,          # istante (1-4) mostrato con la maschera nella terza riga
    print_mask: bool = True,         # True: terza riga con la maschera sull'ottico di flood_instant; False: solo 2 righe
    sar_channel: int = 0,            # canale SAR in scala di grigi (0 = VV, 1 = VH)
    stored_gamma: float = 2.2,       # rgb_gamma con cui e' stato costruito OPT_RGB: serve per tornare ai valori lineari
    display_gamma: float = 2.0,      # gamma in visualizzazione (>1 schiarisce i mezzi toni, 1.0 = lineare)
    display_stretch: bool = True,    # stira il contrasto (percentili 1-99) sui 4 istanti insieme
    display_saturation: float = 1.4, # 1.0 = colori come salvati, >1 piu' vivaci
    dpi: int = 150,                  # risoluzione dei PNG: 150 ~2 MB e ~1 s ciascuno, 100 ~1 MB e ~0.6 s (utile con molte serie)
    job_name: str = "flood_overlay",
    save_dir: str = "/data/flood_overlay",
):
    """
    Per ogni serie campionata salva un PNG con una colonna per istante:
      riga 0 = serie SAR, canale `sar_channel` in scala di grigi (come visual_series)
      riga 1 = serie ottica true color (da OPT_RGB)
      riga 2 = solo con print_mask=True: l'istante `flood_instant` ottico con i pixel flood
               (MASK) in magenta; gli altri riquadri restano bianchi.
    Con print_mask=False il PNG ha solo le prime due righe.
    La maschera e' una sola per serie (annotazione dell'evento).

    I colori dell'ottico sono regolati solo in visualizzazione, senza ricostruire l'artifact: si torna ai
    valori lineari (stored_gamma), si stira il contrasto sui 4 istanti insieme, si applica
    display_gamma e si aumenta la saturazione. Stessa trasformazione per i tre canali e per
    tutti gli istanti, quindi i colori restano coerenti tra un istante e l'altro.

    SAR, OPT_RGB e MASK vengono letti direttamente dagli zip, una serie alla volta, senza estrarli su
    disco: l'uso di disco extra e' solo quello dei PNG (piu' lo zip finale), anche con migliaia di serie.
    """
    import traceback
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(dataset).download("/data/dataset_floods_overlay")
    print(f"OK -> {dataset_path}", flush=True)

    save_dir = save_dir + "_" + job_name
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    # elenco dei partNN.zip di una modalita', senza glob (nome condiviso tra file in src)
    def _list_parts(modality):
        mod_dir = os.path.join(dataset_path, modality)
        if not os.path.isdir(mod_dir):
            print(f"EXC -> cartella '{modality}' non trovata in {dataset_path}, "
                  f"contenuto: {os.listdir(dataset_path)}", flush=True)
            return []
        parts = sorted(
            os.path.join(mod_dir, n) for n in os.listdir(mod_dir)
            if n.startswith("part") and n.endswith(".zip")
        )
        print(f"OK -> {modality}: {len(parts)} part*.zip trovati", flush=True)
        return parts

    def _enhance(rgb_arr):
        """(3, T, H, W) in [0,1] come salvato in OPT_RGB -> stessa forma, solo per la visualizzazione."""
        x = np.clip(rgb_arr, 0.0, 1.0).astype(np.float32)
        valid = x.sum(axis=0) > 0                       # esclude i pixel nulli (dati mancanti, a 0 in tutti i canali)
        x = x ** stored_gamma                           # torno ai valori lineari
        if display_stretch and valid.any():
            lo, hi = np.percentile(x[:, valid], [1, 99])
            if hi - lo > 1e-6:
                x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        x = x ** (1.0 / display_gamma)
        if display_saturation != 1.0:
            lum = (0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2])[None]
            x = np.clip(lum + display_saturation * (x - lum), 0.0, 1.0)
        return x

    def _plot(sar_arr, rgb_arr, flood_mask, ID, n_flood, out_path):
        """
        sar_arr: (T, H, W) in [0,1], gia' ridotto al canale sar_channel
        rgb_arr: (3, T, H, W) in [0,1] gia' regolato per la visualizzazione
        flood_mask: (H, W) bool
        """
        try:
            T = rgb_arr.shape[1]
            t_flood = flood_instant - 1
            if print_mask and not 0 <= t_flood < T:
                raise ValueError(f"flood_instant={flood_instant} fuori da 1..{T}")

            n_rows = 3 if print_mask else 2
            fig, axes = plt.subplots(n_rows, T, figsize=(4 * T, 4 * n_rows), squeeze=False)
            pct = 100.0 * n_flood / flood_mask.size
            title = f"Serie: {ID} | flood: {n_flood} px ({pct:.1f}%)"
            if print_mask:
                title += f", magenta = classe {flood_class}"
            fig.suptitle(title, fontsize=14)

            for t in range(T):
                # riga 0: SAR in scala di grigi
                axes[0, t].imshow(sar_arr[t], cmap="gray", vmin=0, vmax=1)
                axes[0, t].set_title(f"SAR t{t + 1}")
                axes[0, t].axis('off')

                # riga 1: ottico true color
                rgb = np.clip(np.transpose(rgb_arr[:, t], (1, 2, 0)), 0.0, 1.0)     # (H, W, 3)
                axes[1, t].imshow(rgb)
                axes[1, t].set_title(f"OPT t{t + 1}")
                axes[1, t].axis('off')

                # riga 2 (opzionale): maschera sull'ottico di flood_instant, il resto bianco
                if print_mask:
                    if t == t_flood:
                        over = rgb.copy()
                        over[flood_mask] = [1.0, 0.0, 1.0]
                        axes[2, t].imshow(over)
                        axes[2, t].set_title(f"OPT t{t + 1} + flood")
                    axes[2, t].axis('off')

            plt.tight_layout()
            plt.savefig(out_path, bbox_inches='tight', dpi=dpi, facecolor="white")
            plt.close(fig)
            return True
        except Exception as e:
            print(f"EXC -> PNG {ID}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            plt.close('all')
            return False

    # ------------------------------------------------------------------
    # 1) ID presenti in SAR, OPT_RGB e MASK, e pixel flood di ogni maschera
    # ------------------------------------------------------------------
    rgb_ids = set()
    for part_path in _list_parts("OPT_RGB"):
        with zipfile.ZipFile(part_path, 'r') as z:
            rgb_ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
    print(f"{len(rgb_ids)} serie OPT_RGB trovate", flush=True)

    # SAR: indice {ID: (zip aperto, nome)} per leggere una serie alla volta piu' avanti
    sar_handles, sar_index = [], {}
    for part_path in _list_parts("SAR"):
        z = zipfile.ZipFile(part_path, 'r')
        sar_handles.append(z)
        for n in z.namelist():
            if n.endswith('.npy'):
                sar_index[n[:-4]] = (z, n)
    print(f"{len(sar_index)} serie SAR trovate", flush=True)

    flood_counts = {}
    for part_path in _list_parts("MASK"):
        with zipfile.ZipFile(part_path, 'r') as z:
            for n in z.namelist():
                if n.endswith('.npy'):
                    m = np.load(io.BytesIO(z.read(n)))
                    flood_counts[n[:-4]] = int((m == flood_class).sum())
    print(f"{len(flood_counts)} maschere trovate", flush=True)

    common = rgb_ids & set(sar_index) & set(flood_counts)
    candidates = sorted(ID for ID in common if flood_counts[ID] >= max(min_flood_pixels, 0))
    print(f"{len(common)} serie con SAR, OPT_RGB e maschera, {len(candidates)} con >= {min_flood_pixels} px flood", flush=True)

    if not candidates:
        for z in sar_handles:
            z.close()
        raise RuntimeError("Nessuna serie candidata: abbassa min_flood_pixels o controlla che l'artifact "
                           "contenga SAR/, OPT_RGB/ e MASK/")

    sampled = random.sample(candidates, min(n_samples, len(candidates)))
    print(f"ID campionati: {sampled}", flush=True)

    # ------------------------------------------------------------------
    # 2) maschere dei campionati in memoria; OPT_RGB e SAR letti direttamente
    #    dagli zip, una serie alla volta (niente estrazione su disco)
    # ------------------------------------------------------------------
    wanted = {f"{ID}.npy" for ID in sampled}

    masks = {}
    for part_path in _list_parts("MASK"):
        with zipfile.ZipFile(part_path, 'r') as z:
            for n in z.namelist():
                if n in wanted:
                    masks[n[:-4]] = np.load(io.BytesIO(z.read(n)))

    verbose = len(sampled) <= 50          # con molte serie: un log ogni 100
    done, n_fail = set(), 0
    t_start = time.time()

    try:
        for part_path in _list_parts("OPT_RGB"):
            with zipfile.ZipFile(part_path, 'r') as z:
                for n in z.namelist():
                    if n not in wanted:
                        continue
                    ID = n[:-4]
                    if ID not in masks:
                        n_fail += 1
                        print(f"EXC -> {ID}: maschera non trovata, salto", flush=True)
                        continue

                    rgb_arr = _enhance(np.load(io.BytesIO(z.read(n))))     # (3, T, H, W)
                    mask = masks[ID]                                       # (H, W)
                    H, W = rgb_arr.shape[2], rgb_arr.shape[3]

                    # SAR della stessa serie, solo il canale richiesto
                    sar_full = np.load(io.BytesIO(sar_index[ID][0].read(sar_index[ID][1])))   # (C, T, H, W)
                    if not 0 <= sar_channel < sar_full.shape[0]:
                        n_fail += 1
                        print(f"EXC -> {ID}: sar_channel={sar_channel} fuori da 0..{sar_full.shape[0] - 1}", flush=True)
                        continue
                    sar_arr = np.clip(sar_full[sar_channel], 0.0, 1.0)            # (T, H, W)
                    if sar_arr.shape != (rgb_arr.shape[1], H, W):
                        print(f"ATTENZIONE -> {ID}: SAR {sar_arr.shape} diverso da OPT_RGB "
                              f"{(rgb_arr.shape[1], H, W)}", flush=True)

                    if mask.shape != (H, W):
                        print(f"ATTENZIONE -> {ID}: maschera {mask.shape} diversa dall'immagine {(H, W)}, "
                              f"ridimensiono (nearest): controlla che siano davvero allineate", flush=True)
                        rows = (np.arange(H) * mask.shape[0]) // H
                        cols = (np.arange(W) * mask.shape[1]) // W
                        mask = mask[rows][:, cols]

                    flood_mask = (mask == flood_class)
                    n_flood = int(flood_mask.sum())
                    ok = _plot(sar_arr, rgb_arr, flood_mask, ID, n_flood,
                               os.path.join(save_dir, f"SAR_OPT_{ID}_flood.png"))
                    done.add(ID)
                    if not ok:
                        n_fail += 1

                    if verbose:
                        print(f"{'OK' if ok else 'EXC'} -> {ID} | sar {sar_arr.shape} | rgb {rgb_arr.shape} | "
                              f"flood px: {n_flood}", flush=True)
                    elif len(done) % 100 == 0 or len(done) == len(sampled):
                        el = time.time() - t_start
                        print(f"{len(done)}/{len(sampled)} serie | {el / 60:.1f} min trascorsi | "
                              f"~{el / len(done) * (len(sampled) - len(done)) / 60:.1f} min rimanenti | "
                              f"{n_fail} errori", flush=True)
    finally:
        for z in sar_handles:
            z.close()

    missing = [ID for ID in sampled if ID not in done]
    if missing:
        print(f"EXC -> {len(missing)} serie campionate non trovate in OPT_RGB, esempio: {missing[:5]}", flush=True)

    n_png = len([f for f in os.listdir(save_dir) if f.endswith(".png")])
    print(f"PNG prodotti: {n_png}", flush=True)
    if n_png == 0:
        raise RuntimeError("Nessun PNG generato: cerca nel log le righe 'EXC ->' qui sopra.")

    zip_base = save_dir.rstrip("/")
    zip_path = f"{zip_base}.zip"
    shutil.make_archive(zip_base, 'zip', save_dir)
    project_work.log_artifact(
        name=f"PNG_{job_name}_{dataset}",
        kind="artifact",
        source=zip_path,
    )
    print(f"OK -> {zip_path} caricato come artifact ({n_png} PNG)", flush=True)

    return "TERMINATO"



def _write_cache_parts(cache_dir, local_root, modality_label, max_part_bytes):
    """
    IDENTICA a quella di crop_norm_artifact, riusata senza modifiche.
 
    Scrive il contenuto di cache_dir in piu' zip da al massimo max_part_bytes
    (calcolato sulla dimensione grezza dei .npy sorgente, quindi la dimensione
    reale di ogni zip compresso sara' sempre <= a questo budget), salvati sotto
    local_root/{modality_label}/partNN.zip.
 
    I .npy sorgente vengono cancellati via via (streaming) per non raddoppiare
    lo spazio mentre si costruisce lo zip corrente. Le PARTI risultanti pero'
    restano sul disco fino alla fine: servono tutte insieme per il singolo
    upload finale dell'intera cartella come un solo artifact.
    """
    modality_dir = os.path.join(local_root, modality_label)
    os.makedirs(modality_dir, exist_ok=True)
 
    part_num = 1
    current_zip = None
    current_path = None
    current_size = 0
 
    def _open_new_part():
        nonlocal current_zip, current_path, current_size
        current_path = os.path.join(modality_dir, f"part{part_num:02d}.zip")
        current_zip = zipfile.ZipFile(current_path, 'w', zipfile.ZIP_DEFLATED)
        current_size = 0
 
    def _close_part():
        nonlocal current_zip, part_num
        current_zip.close()
        size_gb = os.path.getsize(current_path) / 1e9
        print(f"OK -> {modality_label} parte {part_num} pronta ({size_gb:.2f} GB)", flush=True)
        part_num += 1
 
    _open_new_part()
    for fname in os.listdir(cache_dir):
        full_path = os.path.join(cache_dir, fname)
        fsize = os.path.getsize(full_path)
 
        # se aggiungere questo file sfora il budget, chiudo la parte corrente e ne apro una nuova
        if current_size > 0 and current_size + fsize > max_part_bytes:
            _close_part()
            _open_new_part()
 
        current_zip.write(full_path, arcname=fname)
        current_size += fsize
        os.remove(full_path)  # libera spazio .npy subito, file per file
 
    _close_part()
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"OK -> {modality_label} completato, {part_num - 1} parti scritte localmente", flush=True)
 
 
def _list_zarr_groups(split_dir, modality_label):
    """
    Indicizza gli ID di serie disponibili per una modalita' in uno split,
    senza estrarre nulla. Cerca sia il caso 'un solo tar' (es. S1RTC.tar,
    o S2L2A.tar negli split piccoli come test/val) sia il caso 'diviso in
    piu' parti' (es. S2L2A_1.tar, S2L2A_2.tar, S2L2A_3.tar nello split
    train, troppo grande per un unico file) - stessa struttura interna in
    entrambi i casi: "{MODALITA'}/{ID}_{MODALITA'}.zarr.zip" (un solo file
    per serie, uno ZipStore Zarr, non una cartella con tanti membri).
 
    Niente glob: il nome 'glob' e' condiviso tra i file in src e puo'
    diventare il modulo o la funzione a seconda di quale file e' caricato.
 
    Ritorna (lista di tar handle aperti, {ID: (tar_handle_giusto, [nome
    membro nel suo gruppo zarr])}). Tutti gli handle vanno chiusi dal
    chiamante quando non servono piu'.
    """
    tar_paths = sorted(
        os.path.join(split_dir, n) for n in os.listdir(split_dir)
        if n == f"{modality_label}.tar"
        or (n.startswith(f"{modality_label}_") and n.endswith(".tar"))
    )
 
    pattern = re.compile(r"^[^/]+/(.*?)_" + re.escape(modality_label) + r"\.zarr\.zip$")
    handles = []
    groups = {}
    for tar_path in tar_paths:
        tf = tarfile.open(tar_path, 'r:')
        handles.append(tf)
        for member in tf.getmembers():
            m = pattern.match(member.name)
            if m:
                ID = m.group(1)
                if ID not in groups:
                    groups[ID] = (tf, [])
                groups[ID][1].append(member.name)
 
    return handles, groups
 
 
def _list_mask_members(split_dir):
    """
    Indicizza le maschere di uno split: un .tif per serie, "{ID}_annotation_flood.tif"
    dentro MASK.tar (o MASK_*.tar se fosse diviso in parti).
    Ritorna (handle aperti, {ID: (tar_handle, nome membro)}).
    """
    tar_paths = sorted(
        os.path.join(split_dir, n) for n in os.listdir(split_dir)
        if n == "MASK.tar" or (n.startswith("MASK_") and n.endswith(".tar"))
    )
    handles = []
    members = {}
    suffix = "_annotation_flood.tif"
    for tar_path in tar_paths:
        tf = tarfile.open(tar_path, 'r:')
        handles.append(tf)
        for name in tf.getnames():
            base = os.path.basename(name)
            if base.endswith(suffix):
                members[base[:-len(suffix)]] = (tf, name)
    return handles, members
 
 
def _read_zarr_bands(zip_path, band_names_wanted, band_name_map=None):
    """
    Prima meta' di _load_and_process_zarr (stessa aritmetica, nessuna modifica):
    apre un file .zarr.zip (ZipStore Zarr) con variabile 'bands' (time,band,y,x)
    e coordinata 'band' con i nomi veri delle bande, seleziona i canali richiesti
    PER NOME (non per posizione), mette NaN dove nan_mask segnala pixel non validi
    e dove il valore e' un fill-value tipo -9999.
    Ritorna l'array GREZZO (n_channels, n_images, H, W) float32, con NaN.
    """
    store = zarr.storage.ZipStore(zip_path, mode='r')
    ds = xr.open_zarr(store, consolidated=True)
 
    available = [str(b) for b in ds['band'].values]
    wanted = [band_name_map.get(b, b) for b in band_names_wanted] if band_name_map else list(band_names_wanted)
    missing = [b for b in wanted if b not in available]
    if missing:
        store.close()
        raise ValueError(f"bande mancanti {missing}, disponibili nel file: {available}")
 
    data = ds['bands'].sel(band=wanted).values.astype(np.float32)   # (time, channel, H, W)
    data = np.transpose(data, (1, 0, 2, 3))                          # -> (channel, time, H, W)
 
    if 'nan_mask' in ds:
        invalid = np.asarray(ds['nan_mask'].values)                  # (time, H, W), True = non valido
        invalid = np.broadcast_to(invalid[None, :, :, :], data.shape)
        data = np.where(invalid, np.nan, data)
 
    ds.close()
    store.close()
 
    data = np.where(data < -1000, np.nan, data)   # fill-value tipo -9999 (ottico) trattato come mancante
    return data
 
 
def _center_crop(arr, patch_size):
    """
    Ritaglio centrato sugli ultimi due assi (H, W), identico a Singlemodal_Loader._load_and_process:
    start = max(0, (dim - patch_size) // 2). Se l'immagine e' piu' piccola di patch_size non taglia nulla.
    """
    h, w = arr.shape[-2:]
    start_y = max(0, (h - patch_size) // 2)
    start_x = max(0, (w - patch_size) // 2)
    return arr[..., start_y:start_y + patch_size, start_x:start_x + patch_size]
 
 
def _pad_to(arr, patch_size):
    """Padding a zero in basso e a destra fino a patch_size, come nel loader delle serie normali."""
    h, w = arr.shape[-2:]
    pad_y, pad_x = max(0, patch_size - h), max(0, patch_size - w)
    if pad_y == 0 and pad_x == 0:
        return arr
    widths = [(0, 0)] * (arr.ndim - 2) + [(0, pad_y), (0, pad_x)]
    return np.pad(arr, widths, mode='constant', constant_values=0)
 
 
def _percentile_normalize(data, invalid_as_zero=False):
    """
    Normalizzazione come Singlemodal_Loader._load_and_process delle serie normali:
    percentile 2 e 98 calcolati UNA volta su tutta la serie (canali+istanti insieme),
    (x - p_low) / (p_high - p_low), clip in [0,1]. Ritorna (n_channels, n_images, H, W)
    float32. Non modifica l'array in ingresso.
 
    Unica differenza: i pixel non validi (nan_mask / fill-value) sono NaN, quindi i percentili
    sono nanpercentile (identici a np.percentile quando non ci sono NaN) e alla fine i NaN
    diventano 0. Per i pixel non validi il risultato e' lo stesso delle serie normali, dove un
    pixel a 0 o a un fill-value negativo finisce a 0 dopo il clip.
    invalid_as_zero=True conta i pixel non validi come zeri anche nel calcolo dei percentili
    (utile se nelle serie normali i pixel senza dato valgono 0 e partecipano ai percentili).
    """
    if invalid_as_zero:
        stats_data = np.nan_to_num(data, nan=0.0)
        p_low = np.percentile(stats_data, 2)
        p_high = np.percentile(stats_data, 98)
    else:
        p_low = np.nanpercentile(data, 2)
        p_high = np.nanpercentile(data, 98)
    scale = p_high - p_low if (p_high - p_low) > 1e-6 else 1.0
    data = (data - p_low) / scale
    data = np.nan_to_num(data, nan=0.0)
    data = np.clip(data, 0.0, 1.0).astype(np.float32)
    return data
 
 
def _true_color_rgb(raw_opt, rgb_idx, gain=2.5, gamma=2.2):
    """
    Normalizzazione SOLO per la visualizzazione (cartella OPT_RGB): colori naturali
    come nelle immagini Sentinel-2 true color, invece dei percentili calcolati su
    tutte le 10 bande. Il motivo per cui l'OPT normale e' scuro: p98 su tutti i canali
    e' dominato da NIR/SWIR (B8, B8A, B11, B12), molto piu' luminosi del visibile, e
    il rosso/verde/blu finiscono nella parte bassa della scala.
 
    Qui si usano solo le tre bande visibili, con la stessa trasformazione per tutti
    e tre i canali e per tutti gli istanti (bilanciamento colore e confronto tra
    istanti conservati):
      1) unita': se i valori sono DN (p99 > 20) si divide per 10000 -> riflettanza
      2) punto nero: si toglie il 1 percentile (velo atmosferico / offset)
      3) guadagno fisso `gain` (2.5 e' il valore comune per il true color S2) e clip a 1;
         nuvole e neve saturano a bianco, l'acqua resta scura
      4) gamma (~2.2): la riflettanza e' lineare, mentre uno schermo si aspetta valori
         codificati con gamma; senza questo passo i mezzi toni (vegetazione, suolo)
         restano scuri anche con il guadagno
    raw_opt: (n_channels, T, H, W) grezzo con NaN, come da _read_zarr_bands.
    Ritorna ((3, T, H, W) float32 in [0,1] in ordine R,G,B, dict di statistiche per il log).
    """
    rgb = raw_opt[rgb_idx]                         # (3, T, H, W), copia
    if not np.isfinite(rgb).any():
        return np.zeros(rgb.shape, np.float32), {}
 
    raw_p1, raw_p50, raw_p99 = (float(v) for v in np.nanpercentile(rgb, [1, 50, 99]))
    scale = 10000.0 if raw_p99 > 20 else 1.0
    rgb = rgb / scale
 
    black = float(np.nanpercentile(rgb, 1))
    rgb = np.clip(rgb - black, 0.0, None)
    rgb = np.clip(rgb * gain, 0.0, 1.0)
    if gamma != 1.0:
        rgb = rgb ** (1.0 / gamma)
    rgb = np.nan_to_num(rgb, nan=0.0).astype(np.float32)
 
    stats = {
        "raw_p1": round(raw_p1, 4), "raw_p50": round(raw_p50, 4), "raw_p99": round(raw_p99, 4),
        "unita'": "DN/10000" if scale != 1.0 else "riflettanza",
        "punto_nero": round(black, 4),
        "media_out": round(float(rgb.mean()), 3),
        "saturi_%": round(100.0 * float((rgb >= 0.999).mean()), 2),
    }
    return rgb, stats
 
 
@handler()
def crop_norm_artifact(
    series_list_artifact: str = "nomi_serie_bilanciato_v2",   # artifact .txt con un ID di serie per riga; "" = TUTTE le serie trovate negli split
    anomalies_artifact: str = "Floods_Anomalies_v2",
    splits: list = None,            # None -> ["train", "val", "test"], gli ID vengono cercati in tutte e tre
    output_name: str = "Floods_Anomalies_v2_crop_norm_full",
    max_part_gb: float = 15.0,
    max_series: int = 0,            # 0 = nessun limite; N > 0 = solo le prime N (ordine alfabetico) per una prova veloce
    rgb_gain: float = 2.5,          # guadagno fisso della normalizzazione OPT_RGB
    rgb_gamma: float = 2.2,         # gamma della normalizzazione OPT_RGB (1.0 = lineare, piu' scuro)
    include_mask: bool = True,      # True: aggiunge MASK/ (serve MASK.tar in ogni split); False: solo SAR/, OPT/, OPT_RGB/
    patch_size: int = 256,          # ritaglio centrato + padding a zero, come nel loader delle serie normali
    invalid_as_zero: bool = False,  # True: i pixel non validi contano come zeri nei percentili (vedi _percentile_normalize)
):
    """
    Artifact unico con 3 o 4 cartelle, stesse serie in ognuna (un {ID}.npy per serie, in partNN.zip):
      SAR/      (2, 4, H, W)  float32 [0,1], percentili 2-98 (come sempre)
      OPT/      (10, 4, H, W) float32 [0,1], percentili 2-98 (come sempre)
      OPT_RGB/  (3, 4, H, W)  float32 [0,1], R,G,B (B4,B3,B2) con normalizzazione true color per la visualizzazione
      MASK/     (H, W)        uint8, classi 0 = niente, 1 = acqua, 2 = flood (una per serie);
                solo con include_mask=True
    Con include_mask=False non serve MASK.tar e rasterio non viene nemmeno importato.
 
    SAR/ e OPT/ seguono la stessa pipeline di Singlemodal_Loader._load_and_process delle serie
    normali: ritaglio centrato a patch_size, percentili 2-98 globali sulla serie, clip [0,1],
    padding a zero. Vedi _percentile_normalize per l'unica differenza (pixel non validi = NaN).
    """
 
    project_data = dh.get_project("datasets")
    max_part_bytes = int(max_part_gb * 1024 ** 3)
 
    if splits is None:
        splits = ["train", "val", "test"]
 
    sar_bands = ["vv", "vh"]
    opt_bands_project = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    opt_band_map = {b: ("B0" + b[1:] if len(b) == 2 and b[1].isdigit() else b) for b in opt_bands_project}
    rgb_idx = [opt_bands_project.index(b) for b in ("B4", "B3", "B2")]   # R, G, B -> [2, 1, 0]
 
    local_root = "/data/floods_anomalies_cache_upload"
    shutil.rmtree(local_root, ignore_errors=True)
    os.makedirs(local_root, exist_ok=True)
 
    cache_dirs = {
        "SAR": "/data/cache_anom_SAR",
        "OPT": "/data/cache_anom_OPT",
        "OPT_RGB": "/data/cache_anom_OPT_RGB",
    }
    if include_mask:
        cache_dirs["MASK"] = "/data/cache_anom_MASK"
    for d in cache_dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
 
    all_series = not series_list_artifact or not series_list_artifact.strip()
    if all_series:
        wanted_ids = set()
        print("Nessuna lista: elaboro TUTTE le serie trovate negli split"
              + (f" (al massimo {max_series})" if max_series > 0 else ""), flush=True)
    else:
        print(f"Lista serie richieste: {series_list_artifact}", flush=True)
        ids_path = project_data.get_artifact(series_list_artifact).download("/data/wanted_ids.txt", overwrite=True)
        with open(ids_path, 'r') as f:
            wanted_ids = {line.strip() for line in f if line.strip()}
        if max_series > 0:
            wanted_ids = set(sorted(wanted_ids)[:max_series])
        print(f"OK -> {len(wanted_ids)} serie richieste", flush=True)
 
    print(f"Download: {anomalies_artifact}", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/anomalies")
    print("OK -> Download terminato", flush=True)
 
    remaining = set(wanted_ids)
    processed_ids = set()           # ID gia' tentati: non li rifaccio se compaiono in due split
    n_attempted = 0
    n_ok, n_failed = 0, 0
    n_with_flood, n_shape_mismatch = 0, 0
    extract_tmp = "/data/anomalies_extract_tmp"
 
    try:
        for split in splits:
            if not all_series and not remaining:
                break
            if all_series and max_series > 0 and n_attempted >= max_series:
                break
 
            split_dir = os.path.join(anom_path, split)
            if not os.path.isdir(split_dir):
                print(f"EXC -> split '{split}' non trovato in {anom_path}, salto", flush=True)
                continue
 
            print(f"Indicizzazione split '{split}'...", flush=True)
            handles_s1, groups_s1 = _list_zarr_groups(split_dir, "S1RTC")
            handles_s2, groups_s2 = _list_zarr_groups(split_dir, "S2L2A")
            handles_m, members_m = _list_mask_members(split_dir) if include_mask else ([], {})
            all_handles = handles_s1 + handles_s2 + handles_m
 
            if not groups_s1 or not groups_s2:
                print(f"EXC -> nessun tar S1RTC/S2L2A trovato in '{split}', salto", flush=True)
                for tf in all_handles:
                    tf.close()
                continue
 
            available_here = (set(groups_s1) & set(groups_s2)) - processed_ids
            if not all_series:
                available_here &= remaining
            ids_here = sorted(available_here)
            if all_series and max_series > 0:
                ids_here = ids_here[:max_series - n_attempted]
            n_attempted += len(ids_here)
            print(f"OK -> {len(ids_here)} serie da elaborare in '{split}' "
                  f"({len(handles_s1)} parte/i S1RTC, {len(handles_s2)} parte/i S2L2A"
                  + (f", {len(members_m)} maschere)" if include_mask else ")"), flush=True)
 
            for ID in ids_here:
                try:
                    if include_mask and ID not in members_m:
                        raise ValueError("maschera mancante in MASK.tar di questo split")
 
                    os.makedirs(extract_tmp, exist_ok=True)
                    tf_s1, names_s1 = groups_s1[ID]
                    tf_s2, names_s2 = groups_s2[ID]
                    for name in names_s1:
                        tf_s1.extract(name, extract_tmp)
                    for name in names_s2:
                        tf_s2.extract(name, extract_tmp)
                    if include_mask:
                        tf_m, name_m = members_m[ID]
                        tf_m.extract(name_m, extract_tmp)
 
                    s1_group_path = os.path.join(extract_tmp, "S1RTC", f"{ID}_S1RTC.zarr.zip")
                    s2_group_path = os.path.join(extract_tmp, "S2L2A", f"{ID}_S2L2A.zarr.zip")
 
                    raw_sar = _read_zarr_bands(s1_group_path, sar_bands)
                    raw_opt = _read_zarr_bands(s2_group_path, opt_bands_project, band_name_map=opt_band_map)
                    raw_hw = raw_opt.shape[2:]                      # dimensioni prima del ritaglio
 
                    # stesso ordine delle serie normali: ritaglio -> percentili sul ritaglio -> normalizzo -> padding
                    raw_sar = _center_crop(raw_sar, patch_size)
                    raw_opt = _center_crop(raw_opt, patch_size)
                    invalid_sar = 100.0 * float(np.isnan(raw_sar).mean())
                    invalid_opt = 100.0 * float(np.isnan(raw_opt).mean())
 
                    im_sar = _pad_to(_percentile_normalize(raw_sar, invalid_as_zero), patch_size)
                    im_opt = _pad_to(_percentile_normalize(raw_opt, invalid_as_zero), patch_size)
                    im_rgb, rgb_stats = _true_color_rgb(raw_opt, rgb_idx, gain=rgb_gain, gamma=rgb_gamma)
                    im_rgb = _pad_to(im_rgb, patch_size)
 
                    if include_mask:
                        with rasterio.open(os.path.join(extract_tmp, name_m)) as src:
                            mask = src.read(1)
                        mask = np.clip(mask, 0, 255).astype(np.uint8)
                        if mask.shape != tuple(raw_hw):
                            n_shape_mismatch += 1
                            if n_shape_mismatch <= 5:
                                print(f"ATTENZIONE -> {ID}: maschera {mask.shape} diversa dall'immagine "
                                      f"{tuple(raw_hw)}", flush=True)
                        mask = _pad_to(_center_crop(mask, patch_size), patch_size)
 
                    # scrivo le cartelle solo a serie completata, cosi' gli ID sono sempre gli stessi ovunque
                    np.save(os.path.join(cache_dirs["SAR"], f"{ID}.npy"), im_sar)
                    np.save(os.path.join(cache_dirs["OPT"], f"{ID}.npy"), im_opt)
                    np.save(os.path.join(cache_dirs["OPT_RGB"], f"{ID}.npy"), im_rgb)
                    if include_mask:
                        np.save(os.path.join(cache_dirs["MASK"], f"{ID}.npy"), mask)
                        if (mask == 2).any():
                            n_with_flood += 1
 
                    n_ok += 1
                    if n_ok <= 3 or n_ok % 500 == 0:
                        print(f"  [{n_ok}] {ID} | immagine {tuple(raw_hw)} -> {im_opt.shape[2:]} | pixel non validi: "
                              f"SAR {invalid_sar:.2f}%, OPT {invalid_opt:.2f}% | OPT_RGB {rgb_stats}", flush=True)
                except Exception as e:
                    n_failed += 1
                    print(f"EXC -> serie {ID} ({split}): {e}", flush=True)
                finally:
                    shutil.rmtree(extract_tmp, ignore_errors=True)
 
            processed_ids.update(ids_here)
            remaining -= set(ids_here)
            for tf in all_handles:
                tf.close()
 
        if all_series:
            print(f"OK -> {n_ok} serie processate, {n_failed} fallite", flush=True)
        else:
            print(f"OK -> {n_ok} serie processate, {n_failed} fallite, {len(remaining)} mai trovate in nessuno split", flush=True)
        if include_mask:
            print(f"OK -> {n_with_flood} serie con almeno un pixel flood (classe 2), "
                  f"{n_shape_mismatch} con maschera di forma diversa dall'immagine", flush=True)
        if not all_series and remaining:
            esempio = sorted(remaining)[:10]
            print(f"EXC -> {len(remaining)} ID richiesti non trovati, esempio: {esempio}", flush=True)
 
        if n_ok == 0:
            return "TERMINATO -> nessuna serie processata, controlla i log EXC sopra"
 
        for label in cache_dirs:
            _write_cache_parts(cache_dirs[label], local_root, label, max_part_bytes)
 
        print("Upload dell'intera cartella come artifact unico...", flush=True)
        project_data.log_artifact(
            name=output_name,
            kind='artifact',
            source=local_root,
        )
        print(f"OK -> artifact unico caricato ({output_name})", flush=True)
 
        shutil.rmtree(local_root, ignore_errors=True)
        print("OK -> pulizia locale completata", flush=True)
 
    except Exception as e:
        print(f"EXC -> costruzione cache fallita: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
 
    folders = ", ".join(f"{k}/" for k in cache_dirs)
    flood_info = f"{n_with_flood} con flood, " if include_mask else ""
    return f"TERMINATO -> {n_ok} serie in '{output_name}' ({folders}), {flood_info}{n_failed} fallite"


def save_reconstruction_pngs(model, x, save_dir="reconstruction_img", prefix="series", channels=None, device=None):

    os.makedirs(save_dir, exist_ok=True)
 
    if device is None:
        device = next(model.parameters()).device
 
    x = torch.as_tensor(x).float()
    if x.dim() == 4:                # (C, T, H, W)
        x = x.unsqueeze(0)
    x = x.to(device)
 
    model.train()
    with torch.no_grad():
        out = model(x)
 
    x_np = x[0].cpu().numpy()       # (C, T, H, W)
    out_np = out[0].cpu().numpy()   # (C, T, H, W)
    C, T, H, W = x_np.shape
 
    print(f"INPUT  ({prefix}) - min: {x_np.min():.4f}, max: {x_np.max():.4f}, "
          f"mean: {x_np.mean():.4f}, std: {x_np.std():.4f}", flush=True)
    print(f"OUTPUT ({prefix}) - min: {out_np.min():.4f}, max: {out_np.max():.4f}, "
          f"mean: {out_np.mean():.4f}, std: {out_np.std():.4f}", flush=True)
 
    if channels is None:
        channels = list(range(min(3, C)))
    rgb_mode = len(channels) == 3
 
    for t in range(T):
        in_frame = np.clip(x_np[:, t, :, :][channels], 0.0, 1.0)
        out_frame = np.clip(out_np[:, t, :, :][channels], 0.0, 1.0)
        out_path = os.path.join(save_dir, f"{prefix}_t{t + 1}.png")
 
        if rgb_mode:
            in_img = np.transpose(in_frame, (1, 2, 0))
            out_img = np.transpose(out_frame, (1, 2, 0))
 
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(in_img)
            axes[0].set_title(f"Input t{t + 1}")
            axes[1].imshow(out_img)
            axes[1].set_title(f"Output t{t + 1}")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{prefix} - istante {t + 1}/{T} (canali {channels})")
 
        else:
            n_ch = len(channels)
            fig, axes = plt.subplots(2, n_ch, figsize=(3 * n_ch, 6), squeeze=False)
            for i, ch in enumerate(channels):
                axes[0][i].imshow(in_frame[i], cmap="gray", vmin=0, vmax=1)
                axes[0][i].set_title(f"Input ch{ch}")
                axes[0][i].axis("off")
                axes[1][i].imshow(out_frame[i], cmap="gray", vmin=0, vmax=1)
                axes[1][i].set_title(f"Output ch{ch}")
                axes[1][i].axis("off")
            fig.suptitle(f"{prefix} - istante {t + 1}/{T}")
 
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
 
    print(f"OK -> {T} PNG salvati in {save_dir}/ (prefisso '{prefix}')", flush=True)




@handler()
def debug_ocsvm(
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
    ocsvm_gamma: str = "scale",                     # non usato in debug_ocsvm (resta per compatibilità con la chiamata)
    gamma_mults: str = "1,0.1,0.01,0.001",          # moltiplicatori di gamma="scale", separati da virgola
    standardize: bool = True,                       
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
    normal_fit_ids = shuffled_test_ids[:num_fit_series]                                             # fit -> iperpiano
    normal_test_ids = shuffled_test_ids[num_fit_series:(num_fit_series+len(anom_ids))]              # anomaly detection 
 
 
    print(f"OK -> {len(test_ids)} serie normali tot, {len(normal_fit_ids)} serie fit, {len(normal_test_ids)} serie test normali")
 
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
        print(f"OK -> embedding caricatu: {len(q_fit)} fit, {len(q_normal)} normali, {len(q_anom)} anomale", "\n", flush=True)
 
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
        print("Embedding MoCo: serie di fit ocsvm", flush=True)
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
    w_fit = np.concatenate([q_fit, k_fit], axis=1).astype(np.float64)
    w_normal = np.concatenate([q_normal, k_normal], axis=1).astype(np.float64)
    w_anom = np.concatenate([q_anom, k_anom],axis=1).astype(np.float64)
 
    if standardize:
        scaler = StandardScaler().fit(w_fit)
        w_fit, w_normal, w_anom = scaler.transform(w_fit), scaler.transform(w_normal), scaler.transform(w_anom)

    pw = PCA(whiten=True).fit(w_fit)
    z_fit    = pw.transform(w_fit)
    z_normal = pw.transform(w_normal)
    z_anom   = pw.transform(w_anom)    
 
    mults = [float(m) for m in gamma_mults.split(",")]
    gamma_scale = 1 / (z_fit.shape[1] * z_fit.var())                   # quello che sklearn calcola con gamma="scale"
    print(f"nu={ocsvm_nu}, gamma scale={gamma_scale:.6f}, moltiplicatori={mults}, standardize={standardize}", flush=True)

    # riferimento: Mahalanobis (distanza al quadrato dal centro nello spazio con whitening)
    y = np.r_[np.zeros(len(z_normal)), np.ones(len(z_anom))]
    d2 = (np.r_[z_normal, z_anom] ** 2).sum(axis=1)
    print(f"MD (riferimento): auroc {roc_auc_score(y, d2):.4f}", flush=True)

    # per ogni gamma: auroc, correlazione con la distanza dal centro, frazione fuori dal confine (score < 0)
    for mult in mults:
        ocsvm = OneClassSVM(kernel="rbf", nu=ocsvm_nu, gamma=gamma_scale * mult).fit(z_fit)

        # punteggi: decision_function > 0 dentro il confine , < 0 fuori
        score_fit = ocsvm.decision_function(z_fit).astype(np.float64)
        score_normal = ocsvm.decision_function(z_normal).astype(np.float64)
        score_anom = ocsvm.decision_function(z_anom).astype(np.float64)

        auroc = roc_auc_score(y, -np.r_[score_normal, score_anom])
        corr = np.corrcoef(np.r_[score_normal, score_anom], -d2)[0, 1]
        print(f"gamma = scale x {mult:<7g} | auroc {auroc:.4f} | corr con distanza dal centro {corr:+.2f} | "
              f"fuori dal confine (score < 0): fit {(score_fit < 0).mean():.3f}, normali {(score_normal < 0).mean():.3f}, anomale {(score_anom < 0).mean():.3f}", flush=True)

    return "TERMINATO -> debug ocsvm"    




def _index_parts(root, modality):
    """
    Trova la cartella {modality}/ dentro l'artifact scaricato e indicizza tutte le
    partNN.zip: {ID: (zipfile aperto, nome membro)}. Gli handle vanno chiusi dal chiamante.
    """
    mod_dir = None
    for dirpath, _, _ in os.walk(root):
        if os.path.basename(dirpath) == modality:      # confronto esatto: "OPT" != "OPT_RGB"
            mod_dir = dirpath
            break
    if mod_dir is None:
        raise FileNotFoundError(f"cartella {modality}/ non trovata in {root}")

    handles, index = [], {}
    for n in sorted(os.listdir(mod_dir)):
        if n.endswith(".zip"):
            zf = zipfile.ZipFile(os.path.join(mod_dir, n), 'r')
            handles.append(zf)
            for name in zf.namelist():
                if name.endswith(".npy"):
                    index[os.path.basename(name)[:-4]] = (zf, name)
    return handles, index


def _load_npy(zf, name):
    """Legge un .npy da dentro uno zip direttamente in memoria, senza estrarlo su disco."""
    return np.load(io.BytesIO(zf.read(name)))


def _split_in_4(arr, tile=128):
    """
    (..., 2*tile, 2*tile) -> [k0 alto-sx, k1 alto-dx, k2 basso-sx, k3 basso-dx],
    ognuno (..., tile, tile). Taglia solo gli ultimi due assi: canali e istanti restano intatti.
    """
    h, w = arr.shape[-2:]
    if (h, w) != (2 * tile, 2 * tile):
        raise ValueError(f"forma spaziale {(h, w)}, attesa {(2 * tile, 2 * tile)}")
    return [arr[..., y:y + tile, x:x + tile] for y in (0, tile) for x in (0, tile)]


@handler()
def split_tiles_artifact(
    input_artifact: str = "Floods_Anomalies_v2_crop_norm_full",
    stats_name: str = "Floods_Anomalies_v2_tiles128_stats",   # artifact con il CSV di tutti i tasselli
    output_name: str = "Floods_Anomalies_v2_tiles128",        # artifact con i tasselli selezionati
    tile: int = 128,
    max_part_gb: float = 15.0,
    max_series: int = 0,          # 0 = tutte; N > 0 = solo le prime N serie originali per una prova
    min_flood_pct: float = 5.0,   # tiene i tasselli con flood_pct >= questo valore
    max_per_event: int = 0,       # tetto di tasselli per evento (codice EMSR); 0 = nessun tetto
    seed: int = 0,                # seme del campionamento per evento: stessa selezione a ogni run
):
    """
    Divide ogni serie in 4 nuove serie tile x tile ({ID}_k0..k3), ognuna con tutti gli istanti
    originali, e salva solo quelle selezionate:
      - flood_pct (classe 2 della maschera) >= min_flood_pct
      - al massimo max_per_event per evento, campionate a caso con `seed` (0 = nessun tetto)
    Output (stessa organizzazione dell'artifact di partenza, un {ID}_k{k}.npy per serie in partNN.zip):
      SAR/ (2, 4, tile, tile), OPT/ (10, 4, tile, tile), OPT_RGB/ (3, 4, tile, tile), MASK/ (tile, tile)
    In piu' carica stats_name: CSV con le statistiche di TUTTI i tasselli e la colonna `selected`.
    """
    project_data = dh.get_project("datasets")
    max_part_bytes = int(max_part_gb * 1024 ** 3)
    modalities = ["SAR", "OPT", "OPT_RGB", "MASK"]

    local_root = "/data/tiles_upload"
    shutil.rmtree(local_root, ignore_errors=True)
    os.makedirs(local_root, exist_ok=True)
    cache_dirs = {m: f"/data/cache_tiles_{m}" for m in modalities}
    for d in cache_dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    print(f"Download: {input_artifact}", flush=True)
    in_path = project_data.get_artifact(input_artifact).download("/data/tiles_input", overwrite=True)
    print("OK -> Download terminato", flush=True)

    handles, indexes = [], {}
    rows, selected = [], []
    n_ok, n_failed, n_saved, n_save_failed = 0, 0, 0, 0
    try:
        for m in modalities:
            h, idx = _index_parts(in_path, m)
            handles += h
            indexes[m] = idx

        # --- serie originali ---
        all_ids = set().union(*(set(indexes[m]) for m in modalities))
        print(f"SERIE ORIGINALI -> {len(all_ids)} ID distinti in totale", flush=True)
        for m in modalities:
            print(f"  {m}: {len(indexes[m])} serie", flush=True)

        # --- serie presenti in tutte le cartelle ---
        ids = sorted(set.intersection(*(set(indexes[m]) for m in modalities)))
        print(f"SERIE IN TUTTE LE CARTELLE -> {len(ids)}", flush=True)
        escluse = sorted(all_ids - set(ids))
        if escluse:
            print(f"  {len(escluse)} escluse perche' mancanti in almeno una cartella, esempio: {escluse[:10]}", flush=True)
        if max_series > 0:
            ids = ids[:max_series]
            print(f"  limitate a {len(ids)} per la prova (max_series={max_series})", flush=True)

        # ------------------------------------------------------------------
        # PASSO 1: statistiche di tutti i tasselli (legge solo la maschera)
        # ------------------------------------------------------------------
        for ID in ids:
            try:
                for k, mk in enumerate(_split_in_4(_load_npy(*indexes["MASK"][ID]), tile)):
                    rows.append({
                        "tile_id": f"{ID}_k{k}", "ID": ID, "evento": ID.split("_")[0], "k": k,
                        "flood_pct": round(100.0 * float((mk == 2).mean()), 3),
                        "water_pct": round(100.0 * float((mk == 1).mean()), 3),
                        "selected": False,
                    })
                n_ok += 1
            except Exception as e:
                n_failed += 1
                print(f"EXC -> maschera {ID}: {e}", flush=True)
        print(f"TASSELLI TOTALI -> {len(rows)} da {n_ok} serie ({n_failed} fallite)", flush=True)

        # ------------------------------------------------------------------
        # PASSO 2: selezione (soglia flood + tetto per evento)
        # ------------------------------------------------------------------
        by_event = {}
        for r in rows:
            if r["flood_pct"] >= min_flood_pct:
                by_event.setdefault(r["evento"], []).append(r)
        n_candidates = sum(len(g) for g in by_event.values())

        rng = random.Random(seed)
        n_cut = 0
        for ev in sorted(by_event):                                  # ordine fisso -> campionamento ripetibile
            group = sorted(by_event[ev], key=lambda r: r["tile_id"])
            if max_per_event > 0 and len(group) > max_per_event:
                group = rng.sample(group, max_per_event)
                n_cut += 1
            selected += group
        for r in selected:
            r["selected"] = True

        counts = {}
        for r in selected:
            counts[r["evento"]] = counts.get(r["evento"], 0) + 1
        tetto = max_per_event if max_per_event > 0 else "nessuno"
        print(f"CANDIDATI (flood_pct >= {min_flood_pct}) -> {n_candidates} tasselli, {len(by_event)} eventi", flush=True)
        print(f"SELEZIONATI (tetto {tetto}, {n_cut} eventi tagliati) -> {len(selected)} tasselli, "
              f"{len(counts)} eventi", flush=True)
        if counts:
            top = sorted(counts.items(), key=lambda x: -x[1])
            print(f"  evento piu' grande: {top[0][0]} = {top[0][1]} ({100 * top[0][1] / len(selected):.1f}%), "
                  f"primi 10: {100 * sum(n for _, n in top[:10]) / len(selected):.1f}%", flush=True)

        # ------------------------------------------------------------------
        # PASSO 3: salvataggio dei tasselli selezionati, tutte le modalita'
        # ------------------------------------------------------------------
        sel_by_id = {}
        for r in selected:
            sel_by_id.setdefault(r["ID"], []).append(r["k"])

        for i, ID in enumerate(sorted(sel_by_id), 1):
            try:
                # carico e divido tutto prima di scrivere: se una modalita' fallisce non resta nulla a meta'
                tiles = {m: _split_in_4(_load_npy(*indexes[m][ID]), tile) for m in modalities}
                for k in sorted(sel_by_id[ID]):
                    for m in modalities:
                        np.save(os.path.join(cache_dirs[m], f"{ID}_k{k}.npy"), tiles[m][k])
                    n_saved += 1
                if i <= 3 or i % 500 == 0:
                    print(f"  [{i}/{len(sel_by_id)}] {ID} | k salvati {sorted(sel_by_id[ID])} | "
                          f"forma OPT {tiles['OPT'][0].shape}", flush=True)
            except Exception as e:
                n_save_failed += len(sel_by_id[ID])
                print(f"EXC -> salvataggio {ID}: {e}", flush=True)
    finally:
        for zf in handles:
            zf.close()

    # l'input non serve piu': libero disco prima di creare i nuovi zip
    shutil.rmtree("/data/tiles_input", ignore_errors=True)
    print(f"SERIE FINALI -> {n_saved} tasselli salvati ({n_save_failed} falliti)", flush=True)

    if not rows:
        return "TERMINATO -> nessuna serie divisa: controlla i log EXC sopra"

    # --- CSV con le statistiche di tutti i tasselli + colonna selected ---
    csv_path = "/data/tiles_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    project_data.log_artifact(name=stats_name, kind='artifact', source=csv_path)
    print(f"OK -> CSV caricato come artifact '{stats_name}' ({len(rows)} righe)", flush=True)
    os.remove(csv_path)

    if n_saved == 0:
        return f"TERMINATO -> nessun tassello selezionato/salvato, artifact '{output_name}' non creato"

    # --- artifact con i tasselli selezionati, stessa struttura dell'originale ---
    for label in cache_dirs:
        _write_cache_parts(cache_dirs[label], local_root, label, max_part_bytes)

    print("Upload dell'intera cartella come artifact unico...", flush=True)
    project_data.log_artifact(name=output_name, kind='artifact', source=local_root)
    print(f"OK -> artifact unico caricato ({output_name})", flush=True)
    shutil.rmtree(local_root, ignore_errors=True)

    return (f"TERMINATO -> {n_saved} tasselli salvati in '{output_name}' "
            f"(flood >= {min_flood_pct}%, tetto {tetto}), {len(counts)} eventi, CSV in '{stats_name}'")