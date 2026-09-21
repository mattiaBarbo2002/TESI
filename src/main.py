
import os
import zipfile
import shutil
import torch
import torch.nn as nn
import torch.multiprocessing
import pandas as pd
import numpy as np
import rasterio
import digitalhub as dh
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import json
#from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import time
import zipfile
from glob import glob

import torchvision.transforms as transforms
from digitalhub_runtime_python import handler

from multimodal_3Dconv_attention import Singlemodal_CAE
from multimodal_3Dconv_attention import Singlemodal_CAE_2d
from multimodal_3Dconv_attention import MaskedAutoEncoder

from moco.loader import Singlemodal_Loader
from moco.loader import MoCo2encodersLoader
from moco.loader import PairsLoader

from moco.builder import MoCo2encoders


import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


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


def recalibrate_batchnorm(model, dataloader, num_batches=None, device=None):
    """
    Ricalibra running_mean/running_var di tutti i BatchNorm del modello
    facendo solo forward pass (nessun backward, nessun update dei pesi).
    Al termine il modello resta in eval(), pronto per uso normale.
 
    num_batches=None (default) -> passa sull'intero dataloader una volta,
    copre tutto il dataset passato (consigliato: zero rischio di campione
    non rappresentativo). Passa un numero se il dataset è troppo grande
    per farlo in tempi ragionevoli.
    """
    if device is None:
        device = next(model.parameters()).device
 
    bn_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
 
    old_momentum = []
    for m in bn_layers:
        old_momentum.append(m.momentum)
        m.reset_running_stats()
        m.momentum = None  # media cumulativa vera su tutti i batch visti,
                            # non media mobile esponenziale pesata sui recenti
 
    model.train()
    with torch.no_grad():
        n = 0
        for batch in dataloader:
            if num_batches is not None and n >= num_batches:
                break
            batch = batch.to(device)
            model(batch)
            n += 1
 
    for m, mom in zip(bn_layers, old_momentum):
        m.momentum = mom  # ripristina il momentum originale
 
    model.eval()
    print(f"OK -> BatchNorm ricalibrato su {n} batch", flush=True)




def log_artifact_safe(project, name, source, kind="artifact", retries=3, delay=5):
   
    for attempt in range(1, retries + 1):
        try:
            art = project.log_artifact(name=name, source=source, kind=kind)
            art.refresh()

            if art.status.state == "READY":
                print(f"OK -> Pesi salvati")
                return True
            print(f"Tentativo {attempt}: stato {art.status.state}", flush=True)

        except Exception as e:
            print(f"Tentativo {attempt} fallito: {e}", flush=True)
        time.sleep(delay)
    print(f"EXC -> Impossibile salvare {name}", flush=True)
    return False

# notebook -> encoders

@handler()
def train_autoencoders_1D(
    epochs: int = 1, 
    batch_size: int = 1, 
    lr: float = 1e-4, 
    weight_decay: float = 1e-4,
    patch_size: int = 256,
    n_images1: int = 4,
    n_channels1: int = 2,
    n_images2: int = 4,
    output_dim: int = 10,
    n_channels2: int = 10,
    mamba: bool = False,
    workers: int = 0,
    job_name: str = "nome_job",
    dataset: str = "Test",
    train_sar: bool = True,
    train_opt: bool = True,
    patience: int = 20,
    min_delta: float = 1e-4,
    resume: bool = True,
    time_debug: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    n_gpus = torch.cuda.device_count()
    print(f"GPU: {n_gpus}", flush=True)

    torch.backends.cudnn.benchmark = True
    
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    # download dataset
    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")
    print("OK -> Download terminato")

    try:

        if train_sar:
            # recupero id serie SAR + mappa serie -> zip che la contiene
            s1_dir = os.path.join(dataset_path, "SAR")
            sar_zip_map = {}
            for zip_file in sorted(glob(os.path.join(s1_dir, "SAR_*.zip"))):
                with zipfile.ZipFile(zip_file, 'r') as z:
                    for n in z.namelist():
                        if n.lower().endswith('.tif'):
                            ID = os.path.basename(n).split('_SAR_')[0]
                            sar_zip_map[ID] = zip_file
            train_data_SAR_IDS = list(sar_zip_map.keys())
            print(f"{len(train_data_SAR_IDS)} serie SAR trovate")

        if train_opt:
            # recupero id serie OPT + mappa serie -> zip che la contiene
            s2_dir = os.path.join(dataset_path, "OPT")
            opt_zip_map = {}
            for zip_file in sorted(glob(os.path.join(s2_dir, "OPT_*.zip"))):
                with zipfile.ZipFile(zip_file, 'r') as z:
                    for n in z.namelist():
                        if n.lower().endswith('.tif'):
                            ID = os.path.basename(n).split('_OPT_')[0]
                            opt_zip_map[ID] = zip_file
            train_data_OPT_IDS = list(opt_zip_map.keys())
            print(f"{len(train_data_OPT_IDS)} serie OPT trovate")

    except Exception as e:
        print(f"EXC -> Eccezione recupero liste: {e}", flush=True)    

    # TRAINING SAR
    if train_sar:
    
        print("\n Training S1")
        modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=output_dim, n_images=n_images1, mamba=mamba).to(device)
        if n_gpus > 1:
            modelS1 = nn.DataParallel(modelS1)
        optimizerS1 = torch.optim.Adam(modelS1.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss().to(device)
        scalerS1 = torch.cuda.amp.GradScaler()
        best_loss = float('inf')

        resultsS1 = {'lr': [], 'train_loss': []}

        if resume:
            try:
                w_path = project_work.get_artifact(f"encoder-s1-weights_{job_name}_{dataset}_{epochs}").download("/data")
                state_dict = torch.load(w_path, map_location=device)
                (modelS1.module if n_gpus > 1 else modelS1).load_state_dict(state_dict)
                print("OK -> pesi S1 vecchi caricati", flush=True)
            except Exception as e:
                print(f"EXC -> nessun peso S1 vecchio, inizializzazione casuale: {e}", flush=True)

            try:
                m_path = project_work.get_artifact(f"metrics-s1_{job_name}_{dataset}_{epochs}").download("/data")
                prev_df = pd.read_csv(m_path)
                best_loss = prev_df['train_loss'].min()
                resultsS1 = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
                print(f"OK -> metriche S1 caricate, best_loss={best_loss}", flush=True)
            except Exception as e:
                print(f"EXC -> nessuna metrica S1 vecchia trovata: {e}", flush=True)

        try:
            datasetS1 = Singlemodal_Loader(
                listIDs=train_data_SAR_IDS, root=dataset_path, zip_map=sar_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images1,
                n_channels=n_channels1, data_type='SAR'
            )
            for zip_path in set(sar_zip_map.values()):
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            print("OK -> ZIP SAR eliminati", flush=True)
        except Exception as e:
            print(f"EXC -> Eccezione in Loader S1:", {e}, flush=True)

        # baseline per mse
        try:
            sample_ids = train_data_SAR_IDS[:50]
            mean_accum, n = None, 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetS1.cache_dir, f"{ID}.npy"))
                if mean_accum is None:
                    mean_accum = np.zeros_like(im, dtype=np.float64)
                mean_accum += im
                n += 1
            mean_image = (mean_accum / n).astype(np.float32)

            total_se, total_count = 0.0, 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetS1.cache_dir, f"{ID}.npy"))
                total_se += np.sum((im - mean_image) ** 2)
                total_count += im.size

            print(f"MSE baseline S1: {total_se/total_count:.6f}", flush=True)
        except Exception as e:
            print(f"EXC -> Eccezione calcolo baseline S2: {e}", flush=True)    


        try:
            dataloaderS1 = DataLoader(datasetS1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        except Exception as e:
            print(f"EXC -> Eccezione in DataLoader S1 {e}", flush=True)

        epochs_no_improve = 0

        try:
            for epoch in range(1, epochs + 1):
                modelS1.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS1, mininterval=5.0)

                if time_debug:
                    t_prev = time.time()

                for im in train_bar:
                    if time_debug:
                        torch.cuda.synchronize()
                        t_data = time.time()

                    im = im.to(non_blocking=True, device=device)

                    if time_debug:
                        torch.cuda.synchronize()
                        t_transfer = time.time()

                    with torch.cuda.amp.autocast():
                        output = modelS1(im)
                        loss = loss_fn(output, im)

                    if time_debug:
                        torch.cuda.synchronize()
                        t_forward = time.time()

                    optimizerS1.zero_grad()
                    scalerS1.scale(loss).backward()
                    scalerS1.step(optimizerS1)
                    scalerS1.update()

                    if time_debug:
                        torch.cuda.synchronize()
                        t_backward = time.time()
                        print(f"dati: {t_data-t_prev:.3f}s | trasferimento: {t_transfer-t_data:.3f}s | "
                            f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                        t_prev = time.time()

                    total_num += dataloaderS1.batch_size
                    total_loss += loss.item() * dataloaderS1.batch_size
                    train_bar.set_description(f'S1 Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')

                epoch_loss = total_loss / total_num
                resultsS1['lr'].append(optimizerS1.param_groups[0]['lr'])
                resultsS1['train_loss'].append(epoch_loss)

                if epoch_loss < best_loss - min_delta:
                    state_dict = modelS1.module.state_dict() if n_gpus > 1 else modelS1.state_dict()
                    torch.save(state_dict, f'modelS1_best_{job_name}_{dataset}_{epochs}.pth')
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                    pd.DataFrame(resultsS1).to_csv(f'log_pretrainS1_{job_name}_{dataset}_{epochs}.csv', index_label='epoch')

                    # evitare scadenza token auth 
                    try:
                        dh.refresh_token()
                        print("OK -> refresh token", flush=True)
                    except Exception as e:
                        print(f"EXC -> refresh token: {e}", flush=True)

                    try:
                        project_work = dh.get_project("floods")
                        print("OK -> get project", flush=True)

                    except Exception as e:
                        print(f"EXC -> get project: {e}", flush=True)

                    log_artifact_safe(project_work, f"encoder-s1-weights_{job_name}_{dataset}_{epochs}", f'modelS1_best_{job_name}_{dataset}_{epochs}.pth')

                    try:    
                        project_work.log_artifact(name=f"metrics-s1_{job_name}_{dataset}_{epochs}", source=f'log_pretrainS1_{job_name}_{dataset}_{epochs}.csv', kind='artifact')
                    except Exception as e:
                        print(f"EXC -> upload intermedio S1: {e}", flush=True)

                else:
                    epochs_no_improve += 1
                    print(f"best loss epoch {epoch} (old): {best_loss}", flush=True)

                    if epochs_no_improve >= patience:
                        print(f"Early Stopping S1: epoch {epoch}, best loss: {best_loss}")
                        break

        except Exception as e:
            print(f"EXC -> Eccezione in model train S1: {e}", flush=True)

        best_loss_s1 = best_loss
        print(f"OK -> Terminato training S1, LOSS (MSE): {best_loss_s1}", flush=True)     


        del modelS1
        torch.cuda.empty_cache() 

        try:
            cache_sar_dir = "/data/cache_SAR" 
            if os.path.exists(cache_sar_dir):
                print(f"Pulizia cartella SAR: {cache_sar_dir}", flush=True)
                shutil.rmtree(cache_sar_dir)
                print("OK -> Cartella SAR rimossa", flush=True)
            else:
                print("Nessuna cache SAR trovata da rimuovere.", flush=True)
        except Exception as e:
            print(f"EXC -> Eccezione durante la pulizia della cache SAR: {e}", flush=True)

    # TRAINING OTTICO
    if train_opt:
    
        print("\n Training S2")
        modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=output_dim, n_images=n_images2, mamba=mamba).to(device)
        if n_gpus > 1:
            modelS2 = nn.DataParallel(modelS2)
        optimizerS2 = torch.optim.Adam(modelS2.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss().to(device)
        scalerS2 = torch.cuda.amp.GradScaler()
        best_loss = float('inf')

        resultsS2 = {'lr': [], 'train_loss': []}

        if resume:
            try:
                w_path = project_work.get_artifact(f"encoder-s2-weights_{job_name}_{dataset}_{epochs}").download("/data")
                state_dict = torch.load(w_path, map_location=device)
                (modelS2.module if n_gpus > 1 else modelS2).load_state_dict(state_dict)
                print("OK -> pesi S2 vecchi caricati", flush=True)
            except Exception as e:
                print(f"EXC -> nessun peso S2 vecchio, inizializzazione casuale: {e}", flush=True)

            try:
                m_path = project_work.get_artifact(f"metrics-s2_{job_name}_{dataset}_{epochs}").download("/data")
                prev_df = pd.read_csv(m_path)
                best_loss = prev_df['train_loss'].min()
                resultsS2 = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
                print(f"OK -> metriche S2 caricate, best_loss={best_loss}", flush=True)
            except Exception as e:
                print(f"EXC -> nessuna metrica S2 vecchia trovata: {e}", flush=True)

        try:
            datasetS2 = Singlemodal_Loader(
                listIDs=train_data_OPT_IDS, root=dataset_path, zip_map=opt_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images2,
                n_channels=n_channels2, data_type='OPT'
            )
            for zip_path in set(opt_zip_map.values()):
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            print("OK -> ZIP OPT eliminati", flush=True)
        except Exception as e:
            print(f"EXC -> Eccezione in Loader S2:", {e}, flush=True)

        try:
            sample_ids = train_data_OPT_IDS[:200]
            mean_accum, n = None, 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetS2.cache_dir, f"{ID}.npy"))
                if mean_accum is None:
                    mean_accum = np.zeros_like(im, dtype=np.float64)
                mean_accum += im
                n += 1
            mean_image = (mean_accum / n).astype(np.float32)

            total_se, total_count = 0.0, 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetS2.cache_dir, f"{ID}.npy"))
                total_se += np.sum((im - mean_image) ** 2)
                total_count += im.size

            print(f"MSE baseline S2: {total_se/total_count:.6f}", flush=True)
        except Exception as e:
            print(f"EXC -> Eccezione calcolo baseline S2: {e}", flush=True)

        try:
            dataloaderS2 = DataLoader(datasetS2, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        except Exception as e:
            print(f"EXC -> Eccezione in DataLoader S2 {e}", flush=True)

        epochs_no_improve = 0

        try:
            for epoch in range(1, epochs + 1):
                modelS2.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS2)

                if time_debug:
                    t_prev = time.time()

                for im in train_bar:
                    if time_debug:
                        torch.cuda.synchronize()
                        t_data = time.time()

                    im = im.to(non_blocking=True, device=device)

                    if time_debug:
                        torch.cuda.synchronize()
                        t_transfer = time.time()

                    with torch.cuda.amp.autocast():
                        output = modelS2(im)
                        loss = loss_fn(output, im)

                    if time_debug:
                        torch.cuda.synchronize()
                        t_forward = time.time()

                    optimizerS2.zero_grad()
                    scalerS2.scale(loss).backward()
                    scalerS2.step(optimizerS2)
                    scalerS2.update()

                    if time_debug:
                        torch.cuda.synchronize()
                        t_backward = time.time()
                        print(f"dati: {t_data-t_prev:.3f}s | trasferimento: {t_transfer-t_data:.3f}s | "
                            f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                        t_prev = time.time()

                    total_num += dataloaderS2.batch_size
                    total_loss += loss.item() * dataloaderS2.batch_size
                    train_bar.set_description(f'S2 Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')

                epoch_loss = total_loss / total_num
                resultsS2['lr'].append(optimizerS2.param_groups[0]['lr'])
                resultsS2['train_loss'].append(epoch_loss)

                if epoch_loss < best_loss - min_delta:
                    state_dict = modelS2.module.state_dict() if n_gpus > 1 else modelS2.state_dict()
                    torch.save(state_dict, f'modelS2_best_{job_name}_{dataset}_{epochs}.pth')
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                    pd.DataFrame(resultsS2).to_csv(f'log_pretrainS2_{job_name}_{dataset}_{epochs}.csv', index_label='epoch')

                    try:
                        dh.refresh_token()
                        print("OK -> refresh token", flush=True)
                    except Exception as e:
                        print(f"EXC -> refresh token: {e}", flush=True)

                    try:
                        project_work = dh.get_project("floods")
                        print("OK -> get project", flush=True)
                    except Exception as e:
                        print(f"EXC -> get project: {e}", flush=True)
    
                    log_artifact_safe(project_work, f"encoder-s2-weights_{job_name}_{dataset}_{epochs}", f'modelS2_best_{job_name}_{dataset}_{epochs}.pth')

                    try:
                        project_work.log_artifact(name=f"metrics-s2_{job_name}_{dataset}_{epochs}", source=f'log_pretrainS2_{job_name}_{dataset}_{epochs}.csv', kind='artifact')
                    except Exception as e:
                        print(f"EXC -> upload intermedio S2: {e}", flush=True)

                else:
                    epochs_no_improve += 1
                    print(f"best loss epoch {epoch} (old): {best_loss}", flush=True)

                    if epochs_no_improve >= patience:
                        print(f"Early Stopping S2: epoch {epoch}, best loss: {best_loss}")
                        break

        except Exception as e:
            print(f"EXC -> Eccezione in model train S2: {e}", flush=True)

        best_loss_s2 = best_loss
        del modelS2
        torch.cuda.empty_cache()

        print(f"OK -> Terminato training S2, LOSS (MSE): {best_loss_s2}", flush=True)

        print("OK -> Terminato training S2\n")

        print("RISULTATI")
        if train_sar: print(f"LOSS (MSE) SAR: {best_loss_s1}") 
        if train_opt: print(f"LOSS (MSE) OPT: {best_loss_s2}") 
    
    return "TERMINATO -> training SAR e OPT finito"


@handler()
def train_autoencoders_2D(
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patch_size: int = 256,
    n_images1: int = 4,
    n_channels1: int = 2,
    n_images2: int = 4,
    output_dim: int = 16,
    n_channels2: int = 10,
    mamba: bool = False,
    workers: int = 0,
    job_name: str = "nome_job",
    dataset: str = "Test",
    train_sar: bool = True,
    train_opt: bool = True,
    patience: int = 20,
    min_delta: float = 1e-4,
    ltae: bool = False,
    resume: bool = True,
    time_debug: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
 
    n_gpus = torch.cuda.device_count()
    print(f"GPU: {n_gpus}", flush=True)
 
    torch.backends.cudnn.benchmark = True
 
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    sar_zip_map = {}
    opt_zip_map = {}
    train_data_SAR_IDS = []
    train_data_OPT_IDS = []
   
 
    # prova prima la cache precalcolata: se disponibile, evita del tutto il
    # download del dataset grezzo per la modalita' in questione
    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}_crop_norm").download("/data/dataset_floods")
    print("OK -> Download terminato")


    if train_sar:
        try:
            os.makedirs("/data/cache_SAR", exist_ok=True)
            part_files = sorted(glob(os.path.join(dataset_path, "SAR", "part*.zip")))
            if part_files:
                for part_path in part_files:
                    with zipfile.ZipFile(part_path, 'r') as z:
                        z.extractall("/data/cache_SAR")
                train_data_SAR_IDS = [f[:-4] for f in os.listdir("/data/cache_SAR") if f.endswith('.npy')]
                sar_cache_ready = True
                print(f"OK -> SAR caricato: {len(train_data_SAR_IDS)} serie", flush=True)
        except Exception as e:
            print(f"EXC -> caricamento SAR: {e}", flush=True)

    if train_opt:
        try:
            os.makedirs("/data/cache_OPT", exist_ok=True)
            part_files = sorted(glob(os.path.join(dataset_path, "OPT", "part*.zip")))
            if part_files:
                for part_path in part_files:
                    with zipfile.ZipFile(part_path, 'r') as z:
                        z.extractall("/data/cache_OPT")
                train_data_OPT_IDS = [f[:-4] for f in os.listdir("/data/cache_OPT") if f.endswith('.npy')]
                opt_cache_ready = True
                print(f"OK -> OPT caricato: {len(train_data_OPT_IDS)} serie", flush=True)
        except Exception as e:
            print(f"EXC -> caricamento OPT: {e}", flush=True)
 
    
 
    # TRAINING SAR
    if train_sar:
 
        print("\n Training S1")
        modelS1 = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=output_dim, n_images=n_images1, n_head=8, d_k=8, ltae=ltae).to(device)
        if n_gpus > 1:
            modelS1 = nn.DataParallel(modelS1)
        optimizerS1 = torch.optim.Adam(modelS1.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss().to(device)
        scalerS1 = torch.cuda.amp.GradScaler()
        best_loss = float('inf')
 
        resultsS1 = {'lr': [], 'train_loss': []}
 
        if resume:
            try:
                w_path = project_work.get_artifact(f"encoder-s1-weights_{job_name}_{dataset}_{epochs}").download("/data")
                state_dict = torch.load(w_path, map_location=device)
                (modelS1.module if n_gpus > 1 else modelS1).load_state_dict(state_dict)
                print("OK -> pesi S1 vecchi caricati", flush=True)
            except Exception as e:
                print(f"EXC -> nessun peso S1 vecchio, inizializzazione casuale: {e}", flush=True)
 
            try:
                m_path = project_work.get_artifact(f"metrics-s1_{job_name}_{dataset}_{epochs}").download("/data")
                prev_df = pd.read_csv(m_path)
                best_loss = prev_df['train_loss'].min()
                resultsS1 = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
                print(f"OK -> metriche S1 caricate, best_loss={best_loss}", flush=True)
            except Exception as e:
                print(f"EXC -> nessuna metrica S1 vecchia trovata: {e}", flush=True)
 
        try:
            datasetS1 = Singlemodal_Loader(
                listIDs=train_data_SAR_IDS, root=dataset_path, zip_map=sar_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images1,
                n_channels=n_channels1, data_type='SAR'
            )
            for zip_path in set(sar_zip_map.values()):
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            print("OK -> ZIP SAR eliminati", flush=True)
        except Exception as e:
            print(f"EXC -> Eccezione in Loader S1:", {e}, flush=True)
 
        # baseline per mse
        try:
            sample_ids = train_data_SAR_IDS[:50]
            mean_accum, n = None, 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetS1.cache_dir, f"{ID}.npy"))
                if mean_accum is None:
                    mean_accum = np.zeros_like(im, dtype=np.float64)
                mean_accum += im
                n += 1
            mean_image = (mean_accum / n).astype(np.float32)
 
            total_se, total_count = 0.0, 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetS1.cache_dir, f"{ID}.npy"))
                total_se += np.sum((im - mean_image) ** 2)
                total_count += im.size
 
            print(f"MSE baseline S1: {total_se/total_count:.6f}", flush=True)
        except Exception as e:
            print(f"EXC -> Eccezione calcolo baseline S2: {e}", flush=True)
 
        try:
            dataloaderS1 = DataLoader(datasetS1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        except Exception as e:
            print(f"EXC -> Eccezione in DataLoader S1 {e}", flush=True)
 
        epochs_no_improve = 0
 
        try:
            for epoch in range(1, epochs + 1):
                modelS1.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS1, mininterval=5.0)
 
                if time_debug:
                    t_prev = time.time()
 
                for im in train_bar:
                    if time_debug:
                        torch.cuda.synchronize()
                        t_data = time.time()
 
                    im = im.to(non_blocking=True, device=device)
 
                    if time_debug:
                        torch.cuda.synchronize()
                        t_transfer = time.time()
 
                    with torch.cuda.amp.autocast():
                        output = modelS1(im)
                        loss = loss_fn(output, im)
 
                    if time_debug:
                        torch.cuda.synchronize()
                        t_forward = time.time()
 
                    optimizerS1.zero_grad()
                    scalerS1.scale(loss).backward()
                    scalerS1.step(optimizerS1)
                    scalerS1.update()
 
                    if time_debug:
                        torch.cuda.synchronize()
                        t_backward = time.time()
                        print(f"dati: {t_data-t_prev:.3f}s | trasferimento: {t_transfer-t_data:.3f}s | "
                            f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                        t_prev = time.time()
 
                    total_num += dataloaderS1.batch_size
                    total_loss += loss.item() * dataloaderS1.batch_size
                    train_bar.set_description(f'S1 Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')
 
                epoch_loss = total_loss / total_num
                resultsS1['lr'].append(optimizerS1.param_groups[0]['lr'])
                resultsS1['train_loss'].append(epoch_loss)
 
                # salva/carica il csv OGNI epoca, non solo sui miglioramenti
                pd.DataFrame(resultsS1).to_csv(f'log_pretrainS1_{job_name}_{dataset}_{epochs}.csv', index_label='epoch')
                try:
                    project_work.log_artifact(name=f"metrics-s1_{job_name}_{dataset}_{epochs}", source=f'log_pretrainS1_{job_name}_{dataset}_{epochs}.csv', kind='artifact')
                except Exception as e:
                    print(f"EXC -> upload intermedio S1: {e}", flush=True)
 
                if epoch_loss < best_loss - min_delta:
                    state_dict = modelS1.module.state_dict() if n_gpus > 1 else modelS1.state_dict()
                    torch.save(state_dict, f'modelS1_best_{job_name}_{dataset}_{epochs}.pth')
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)
 
                    # evitare scadenza token auth
                    try:
                        dh.refresh_token()
                        print("OK -> refresh token")
                    except Exception as e:
                        print(f"EXC -> refresh token: {e}", flush=True)
 
                    try:
                        project_work = dh.get_project("floods")
                        print("OK -> get project")
                    except Exception as e:
                        print(f"EXC -> get project: {e}", flush=True)
 
                    log_artifact_safe(project_work, f"encoder-s1-weights_{job_name}_{dataset}_{epochs}", f'modelS1_best_{job_name}_{dataset}_{epochs}.pth')
 
                else:
                    epochs_no_improve += 1
                    print(f"best loss epoch {epoch} (old): {best_loss}", flush=True)
 
                    if epochs_no_improve >= patience:
                        print(f"Early Stopping S1: epoch {epoch}, best loss: {best_loss}")
                        break
 
        except Exception as e:
            print(f"EXC -> Eccezione in model train S1: {e}", flush=True)
 
        best_loss_s1 = best_loss
        print(f"OK -> Terminato training S1, LOSS (MSE): {best_loss_s1}", flush=True)
 
        del modelS1
        torch.cuda.empty_cache()
 
        try:
            cache_sar_dir = "/data/cache_SAR"
            if os.path.exists(cache_sar_dir):
                print(f"Pulizia cartella SAR: {cache_sar_dir}", flush=True)
                shutil.rmtree(cache_sar_dir)
                print("OK -> Cartella SAR rimossa", flush=True)
            else:
                print("Nessuna cache SAR trovata da rimuovere.", flush=True)
        except Exception as e:
            print(f"EXC -> Eccezione durante la pulizia della cache SAR: {e}", flush=True)
 
    # TRAINING OTTICO
    if train_opt:
 
        print("\n Training S2")
        modelS2 = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=output_dim, n_images=n_images2, n_head=8, d_k=8, ltae=ltae).to(device)
        if n_gpus > 1:
            modelS2 = nn.DataParallel(modelS2)
        optimizerS2 = torch.optim.Adam(modelS2.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss().to(device)
        scalerS2 = torch.cuda.amp.GradScaler()
        best_loss = float('inf')
 
        resultsS2 = {'lr': [], 'train_loss': []}
 
        if resume:
            try:
                w_path = project_work.get_artifact(f"encoder-s2-weights_{job_name}_{dataset}_{epochs}").download("/data")
                state_dict = torch.load(w_path, map_location=device)
                (modelS2.module if n_gpus > 1 else modelS2).load_state_dict(state_dict)
                print("OK -> pesi S2 vecchi caricati", flush=True)
            except Exception as e:
                print(f"EXC -> nessun peso S2 vecchio, inizializzazione casuale: {e}", flush=True)
 
            try:
                m_path = project_work.get_artifact(f"metrics-s2_{job_name}_{dataset}_{epochs}").download("/data")
                prev_df = pd.read_csv(m_path)
                best_loss = prev_df['train_loss'].min()
                resultsS2 = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
                print(f"OK -> metriche S2 caricate, best_loss={best_loss}", flush=True)
            except Exception as e:
                print(f"EXC -> nessuna metrica S2 vecchia trovata: {e}", flush=True)
 
        try:
            datasetS2 = Singlemodal_Loader(
                listIDs=train_data_OPT_IDS, root=dataset_path, zip_map=opt_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images2,
                n_channels=n_channels2, data_type='OPT'
            )
            for zip_path in set(opt_zip_map.values()):
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            print("OK -> ZIP OPT eliminati", flush=True)
        except Exception as e:
            print(f"EXC -> Eccezione in Loader S2:", {e}, flush=True)
 
        try:
            sample_ids = train_data_OPT_IDS[:200]
            mean_accum, n = None, 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetS2.cache_dir, f"{ID}.npy"))
                if mean_accum is None:
                    mean_accum = np.zeros_like(im, dtype=np.float64)
                mean_accum += im
                n += 1
            mean_image = (mean_accum / n).astype(np.float32)
 
            total_se, total_count = 0.0, 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetS2.cache_dir, f"{ID}.npy"))
                total_se += np.sum((im - mean_image) ** 2)
                total_count += im.size
 
            print(f"MSE baseline S2: {total_se/total_count:.6f}", flush=True)
        except Exception as e:
            print(f"EXC -> Eccezione calcolo baseline S2: {e}", flush=True)
 
        try:
            dataloaderS2 = DataLoader(datasetS2, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        except Exception as e:
            print(f"EXC -> Eccezione in DataLoader S2 {e}", flush=True)
 
        epochs_no_improve = 0
 
        try:
            for epoch in range(1, epochs + 1):
                modelS2.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS2)
 
                if time_debug:
                    t_prev = time.time()
 
                for im in train_bar:
                    if time_debug:
                        torch.cuda.synchronize()
                        t_data = time.time()
 
                    im = im.to(non_blocking=True, device=device)
 
                    if time_debug:
                        torch.cuda.synchronize()
                        t_transfer = time.time()
 
                    with torch.cuda.amp.autocast():
                        output = modelS2(im)
                        loss = loss_fn(output, im)
 
                    if time_debug:
                        torch.cuda.synchronize()
                        t_forward = time.time()
 
                    optimizerS2.zero_grad()
                    scalerS2.scale(loss).backward()
                    scalerS2.step(optimizerS2)
                    scalerS2.update()
 
                    if time_debug:
                        torch.cuda.synchronize()
                        t_backward = time.time()
                        print(f"dati: {t_data-t_prev:.3f}s | trasferimento: {t_transfer-t_data:.3f}s | "
                            f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                        t_prev = time.time()
 
                    total_num += dataloaderS2.batch_size
                    total_loss += loss.item() * dataloaderS2.batch_size
                    train_bar.set_description(f'S2 Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')
 
                epoch_loss = total_loss / total_num
                resultsS2['lr'].append(optimizerS2.param_groups[0]['lr'])
                resultsS2['train_loss'].append(epoch_loss)
 
                pd.DataFrame(resultsS2).to_csv(f'log_pretrainS2_{job_name}_{dataset}_{epochs}.csv', index_label='epoch')
                try:
                    project_work.log_artifact(name=f"metrics-s2_{job_name}_{dataset}_{epochs}", source=f'log_pretrainS2_{job_name}_{dataset}_{epochs}.csv', kind='artifact')
                except Exception as e:
                    print(f"EXC -> upload intermedio S2: {e}", flush=True)
 
                if epoch_loss < best_loss - min_delta:
                    state_dict = modelS2.module.state_dict() if n_gpus > 1 else modelS2.state_dict()
                    torch.save(state_dict, f'modelS2_best_{job_name}_{dataset}_{epochs}.pth')
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)
 
                    try:
                        dh.refresh_token()
                        print("OK -> refresh token")
                    except Exception as e:
                        print(f"EXC -> refresh token: {e}", flush=True)
 
                    try:
                        project_work = dh.get_project("floods")
                        print("OK -> get project")
                    except Exception as e:
                        print(f"EXC -> get project: {e}", flush=True)
 
                    log_artifact_safe(project_work, f"encoder-s2-weights_{job_name}_{dataset}_{epochs}", f'modelS2_best_{job_name}_{dataset}_{epochs}.pth')
 
                else:
                    epochs_no_improve += 1
                    print(f"best loss epoch {epoch} (old): {best_loss}", flush=True)
 
                    if epochs_no_improve >= patience:
                        print(f"Early Stopping S2: epoch {epoch}, best loss: {best_loss}")
                        break
 
        except Exception as e:
            print(f"EXC -> Eccezione in model train S2: {e}", flush=True)
 
        best_loss_s2 = best_loss
        del modelS2
        torch.cuda.empty_cache()
 
        print(f"OK -> Terminato training S2, LOSS (MSE): {best_loss_s2}", flush=True)
 
        print("OK -> Terminato training S2\n")
 
        print("RISULTATI")
        if train_sar: print(f"LOSS (MSE) SAR: {best_loss_s1}")
        if train_opt: print(f"LOSS (MSE) OPT: {best_loss_s2}")
 
    return "TERMINATO -> training SAR e OPT finito"



# notebook -> moco
# valori uguali al codice originale tranne coda moco_k

handler()
def train_moco(
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    moco_dim: int = 128,
    moco_k: int = 1024,
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    symmetric: bool = False,
    mamba: bool = False,
    workers: int = 0,
    job_name: str = "nome_job",
    dataset: str = "Test",
    weights_encoder_sar: str = "train_sar_1D_v1_Standard_200",
    weights_encoder_opt: str = "train_opt_1D_v1_Standard_200",
    patience: int = 20,
    min_delta: float = 1e-4,
    calib_batch_size: int = 32,
    time_debug: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    print(f"Download dataset: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")
    print("OK -> Download terminato")

    # creazione lista immagini SAR e OPT
    try:
        s1_dir = os.path.join(dataset_path, "SAR")
        sar_zip_map = {}
        for zip_file in sorted(glob(os.path.join(s1_dir, "SAR_*.zip"))):
            with zipfile.ZipFile(zip_file, 'r') as z:
                for n in z.namelist():
                    if n.lower().endswith('.tif'):
                        ID = os.path.basename(n).split('_SAR_')[0]
                        sar_zip_map[ID] = zip_file
        print(f"{len(sar_zip_map)} serie SAR trovate")

        s2_dir = os.path.join(dataset_path, "OPT")
        opt_zip_map = {}
        for zip_file in sorted(glob(os.path.join(s2_dir, "OPT_*.zip"))):
            with zipfile.ZipFile(zip_file, 'r') as z:
                for n in z.namelist():
                    if n.lower().endswith('.tif'):
                        ID = os.path.basename(n).split('_OPT_')[0]
                        opt_zip_map[ID] = zip_file
        print(f"{len(opt_zip_map)} serie OPT trovate")

        # ordinamento, nel 80% train devo avere stesse serie SAR e OPT
        train_data_IDS = sorted(set(sar_zip_map) & set(opt_zip_map))
        print(f"{len(train_data_IDS)} serie complete SAR e OPT", flush=True)

        # split 80/20
        random.seed(42)
        shuffled_ids = train_data_IDS.copy()
        random.shuffle(shuffled_ids)
        split_idx = int(len(shuffled_ids) * 0.8)
        train_ids = shuffled_ids[:split_idx]
        held_out_ids = shuffled_ids[split_idx:]
        print(f"{len(train_ids)} serie dataset train (80%)", flush=True)
        print(f"{len(held_out_ids)} serie dataset test (20%)", flush=True)

    except Exception as e:
        print(f"EXC -> Eccezione recupero liste: {e}", flush=True)

    # salvataggio log liste train e test
    with open(f'train_ids_{job_name}_{dataset}_{epochs}.json', 'w') as f:
        json.dump(train_ids, f)
    try:
        project_work.log_artifact(name=f"moco-train-ids_{job_name}_{dataset}_{epochs}", source=f'train_ids_{job_name}_{dataset}_{epochs}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload train ids fallito: {e}", flush=True)

    with open(f'test_ids_{job_name}_{dataset}_{epochs}.json', 'w') as f:
        json.dump(held_out_ids, f)
    try:
        project_work.log_artifact(name=f"moco-test-ids_{job_name}_{dataset}_{epochs}", source=f'test_ids_{job_name}_{dataset}_{epochs}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload held-out ids fallito: {e}", flush=True)

    # caricamento pesi encoders
    print("Caricamento pesi Encoders", flush=True)
    path_s1 = project_work.get_artifact(f"encoder-s1-weights_{weights_encoder_sar}").download(f"modelS1_best_{weights_encoder_sar}.pth")
    path_s2 = project_work.get_artifact(f"encoder-s2-weights_{weights_encoder_opt}").download(f"modelS2_best_{weights_encoder_opt}.pth")

    modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=512, n_images=n_images1, mamba=mamba).to(device)
    modelS1.load_state_dict(torch.load(path_s1, map_location=device))

    modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=512, n_images=n_images2, mamba=mamba).to(device)
    modelS2.load_state_dict(torch.load(path_s2, map_location=device))
    print("OK -> Pesi Encoders caricati", flush=True)

    # ricalibrazione batchNorm
    try:
        calib_ids_sar = list(sar_zip_map.keys())
        if len(calib_ids_sar) > 2000:
            calib_ids_sar = random.sample(calib_ids_sar, 2000)

        calib_datasetS1 = Singlemodal_Loader(
            listIDs=calib_ids_sar, root=dataset_path, zip_map=sar_zip_map,
            transform=None, patch_size=patch_size, n_images=n_images1,
            n_channels=n_channels1, data_type='SAR'
        )
        calib_loaderS1 = torch.utils.data.DataLoader(
            calib_datasetS1, batch_size=calib_batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True
        )
        recalibrate_batchnorm(modelS1, calib_loaderS1, num_batches=None, device=device)
        print("OK -> ricalibrazione BN SAR", flush=True)
    except Exception as e:
        print(f"EXC -> ricalibrazione BN SAR: {e}", flush=True)
 
    try:
        calib_ids_opt = list(opt_zip_map.keys())
        if len(calib_ids_opt) > 2000:
            calib_ids_opt = random.sample(calib_ids_opt, 2000)

        calib_datasetS2 = Singlemodal_Loader(
            listIDs=calib_ids_opt, root=dataset_path, zip_map=opt_zip_map,
            transform=None, patch_size=patch_size, n_images=n_images2,
            n_channels=n_channels2, data_type='OPT'
        )
        calib_loaderS2 = torch.utils.data.DataLoader(
            calib_datasetS2, batch_size=calib_batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True
        )
        recalibrate_batchnorm(modelS2, calib_loaderS2, num_batches=None, device=device)
        print("OK -> ricalibrazione BN OPT", flush=True)
    except Exception as e:
        print(f"EXC -> ricalibrazione BN OPT: {e}", flush=True)


    model = MoCo2encoders(
        base_encoder_q=modelS2.encoder,
        base_encoder_k=modelS1.encoder,
        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
        symmetric=symmetric, device=device,
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    try:
        train_dataset = MoCo2encodersLoader(
            listIDs=train_ids,
            sar_map=sar_zip_map,
            opt_map=opt_zip_map,
            transform=None,
            patch_size=patch_size,
            n_images1=n_images1, n_channels1=n_channels1,
            n_images2=n_images2, n_channels2=n_channels2,
        )
    except Exception as e:
        print(f"EXC -> Eccezione MoCo2encodersLoader: {e}", flush=True)

    try:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True,
        )
    except Exception as e:
        print(f"EXC -> Eccezione DataLoader: {e}", flush=True)

    results = {'lr': [], 'train_loss': []}
    best_loss = float('inf')
    epochs_no_improve = 0

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            model.encoder_q.eval()
            model.encoder_k.eval()

            total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)

            if time_debug:
                t_prev = time.time()

            for im_q, im_k in train_bar:
                if time_debug:
                    torch.cuda.synchronize()
                    t_data = time.time()

                im_q = im_q.to(non_blocking=True, device=device)
                im_k = im_k.to(non_blocking=True, device=device)

                # --- INIZIO BLOCCO DI DEBUG ---
                if epoch == 1 and total_num == 0:
                    print("\n" + "="*40, flush=True)
                    print("DIAGNOSTICA STEP 0 - CONTROLLO COLLASSO", flush=True)
                    
                    # 1. Controllo Dataloader (Input)
                    print(f"INPUT im_q (SAR) - min: {im_q.min().item():.4f}, max: {im_q.max().item():.4f}, std: {im_q.float().std().item():.4f}", flush=True)
                    print(f"INPUT im_k (OPT) - min: {im_k.min().item():.4f}, max: {im_k.max().item():.4f}, std: {im_k.float().std().item():.4f}", flush=True)
                    
                    # 2. Controllo Encoder (Output pre-MLP)
                    with torch.no_grad():
                        out_q = model.encoder_q(im_q)
                        out_k = model.encoder_k(im_k)
                        
                    print(f"ENCODER q_out - std: {out_q.std().item():.6f}, val unici: {len(torch.unique(out_q))}", flush=True)
                    print(f"ENCODER k_out - std: {out_k.std().item():.6f}, val unici: {len(torch.unique(out_k))}", flush=True)
                    print("="*40 + "\n", flush=True)
                # --- FINE BLOCCO DI DEBUG ---

                if time_debug:
                    torch.cuda.synchronize()
                    t_transfer = time.time()

                with torch.cuda.amp.autocast():
                    loss = model(im_q, im_k)

                if time_debug:
                    torch.cuda.synchronize()
                    t_forward = time.time()    

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if time_debug:
                    torch.cuda.synchronize()
                    t_backward = time.time()
                    print(f"dati: {t_data-t_prev:.3f}s | trasferimento: {t_transfer-t_data:.3f}s | "
                        f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                    t_prev = time.time()

                    # qui forward comprende tempo passaggio avanti, aggiornamento momentum, calcolo loss, inserimento in coda 

                total_num += batch_size
                total_loss += loss.item() * batch_size
                train_bar.set_description(f'MoCo Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')

            epoch_loss = total_loss / total_num
            results['lr'].append(optimizer.param_groups[0]['lr'])
            results['train_loss'].append(epoch_loss)

            pd.DataFrame(results).to_csv(f'log_moco_{job_name}_{dataset}_{epochs}.csv', index_label='epoch')

            try:
                project_work.log_artifact(name=f"moco-metrics_{job_name}_{dataset}_{epochs}", source=f'log_moco_{job_name}_{dataset}_{epochs}.csv', kind='artifact')
                print(f"OK -> metriche salvate", flush=True)
            except Exception as e:
                print(f"EXC -> upload metriche MoCo: {e}", flush=True)

            if epoch_loss < best_loss - min_delta:
                torch.save(model.state_dict(), f'moco_best_{job_name}_{dataset}_{epochs}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0
                try:
                    project_work.log_artifact(name=f"moco-weights_{job_name}_{dataset}_{epochs}", source=f'moco_best_{job_name}_{dataset}_{epochs}.pth', kind='artifact')
                    print(f"OK -> pesi salvati", flush=True)
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", flush=True)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early Stopping MoCo: epoch {epoch}")
                    break

        best_loss_moco = best_loss

    except Exception as e:
        print(f"EXC -> Eccezione in MoCo train: {e}", flush=True)

    print(f"OK -> training MoCo finito, LOSS (InfoNCE): {best_loss_moco}", flush=True)
    
    return "TERMINATO -> MoCo training finito"


@handler()
def train_mae(
    epochs: int = 1, 
    batch_size: int = 1, 
    lr: float = 1e-4, 
    weight_decay: float = 1e-4,
    patch_size: int = 256,
    n_images1: int = 4,
    n_channels1: int = 2,
    n_images2: int = 4,
    n_channels2: int = 10,
    mamba: bool = False,
    workers: int = 0,
    job_name: str = "nome_job",
    dataset: str = "Test",
    sar_to_opt: bool = True,
    mask: bool = False,
    mask_precentage: float = 0.8,
    mask_square_size: int = 16,
    patience: int = 20,
    min_delta: float = 1e-4,
    time_debug: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, flush=True)

    n_gpus = torch.cuda.device_count()
    print(f"GPU: {n_gpus}\n", flush=True)

    torch.backends.cudnn.benchmark = True
    
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    # download dataset
    print(f"Download dataset: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")
    print("OK -> Download terminato\n")

    try:
        # recupero id serie SAR + mappa serie -> zip che la contiene
        s1_dir = os.path.join(dataset_path, "SAR")
        sar_zip_map = {}
        for zip_file in sorted(glob(os.path.join(s1_dir, "SAR_*.zip"))):
            with zipfile.ZipFile(zip_file, 'r') as z:
                for n in z.namelist():
                    if n.lower().endswith('.tif'):
                        ID = os.path.basename(n).split('_SAR_')[0]
                        sar_zip_map[ID] = zip_file
        train_data_SAR_IDS = list(sar_zip_map.keys())
        print(f"{len(train_data_SAR_IDS)} serie SAR trovate")

    
        # recupero id serie OPT + mappa serie -> zip che la contiene
        s2_dir = os.path.join(dataset_path, "OPT")
        opt_zip_map = {}
        for zip_file in sorted(glob(os.path.join(s2_dir, "OPT_*.zip"))):
            with zipfile.ZipFile(zip_file, 'r') as z:
                for n in z.namelist():
                    if n.lower().endswith('.tif'):
                        ID = os.path.basename(n).split('_OPT_')[0]
                        opt_zip_map[ID] = zip_file
        train_data_OPT_IDS = list(opt_zip_map.keys())
        print(f"{len(train_data_OPT_IDS)} serie OPT trovate")

    except Exception as e:
        print(f"EXC -> Eccezione recupero liste: {e}", flush=True)  

    # Parametri 
    direction = 'sar_to_opt' if sar_to_opt else 'opt_to_sar'
    zip_map_input = sar_zip_map if sar_to_opt else opt_zip_map
    zip_map_target = opt_zip_map if sar_to_opt else sar_zip_map
    input_list = train_data_SAR_IDS if sar_to_opt else train_data_OPT_IDS
    target_list = train_data_OPT_IDS if sar_to_opt else train_data_SAR_IDS
    
    # SAR to OPT
    print(f"\n Training {'sar_to_opt' if sar_to_opt else 'opt_to_sar'}")

    model_mae = MaskedAutoEncoder(
        input_dim = n_channels1 if sar_to_opt else n_channels2, 
        output_channels = n_channels2 if sar_to_opt else n_channels1, 
        output_dim=10, 
        n_images= n_images1 if sar_to_opt else n_images2, 
        mamba=False
    ).to(device)

    if n_gpus > 1:
        model_mae = nn.DataParallel(model_mae)
    optimizer_mae = torch.optim.Adam(model_mae.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss().to(device)
    scaler_mae = torch.cuda.amp.GradScaler()
    best_loss = 9999.0

    input_dim = n_channels1 if sar_to_opt else n_channels2
    try:
        # cache input
        dataset_input = Singlemodal_Loader(
            listIDs = input_list, 
            root = dataset_path, 
            zip_map = sar_zip_map if sar_to_opt else opt_zip_map,
            transform=None, 
            patch_size = patch_size, 
            n_images = n_images1 if sar_to_opt else n_images2,
            n_channels = n_channels1 if sar_to_opt else n_channels2, 
            data_type = 'SAR' if sar_to_opt else 'OPT'
        )
        for zip_path in set(zip_map_input.values()):
            if os.path.exists(zip_path):
                os.remove(zip_path)
        print(f"OK -> ZIP {'SAR' if sar_to_opt else 'OPT'} eliminati\n", flush=True)

        # cache target
        dataset_target = Singlemodal_Loader(
            listIDs = target_list, 
            root = dataset_path, 
            zip_map = opt_zip_map if sar_to_opt else sar_zip_map,
            transform = None, 
            patch_size = patch_size, 
            n_images = n_images2 if sar_to_opt else n_images1,
            n_channels = n_channels2 if sar_to_opt else n_channels1, 
            data_type = 'OPT' if sar_to_opt else 'SAR'
        )
        for zip_path in set(zip_map_target.values()):
            if os.path.exists(zip_path):
                os.remove(zip_path)
        print(f"OK -> ZIP {'OPT' if sar_to_opt else 'SAR'} eliminati\n", flush=True)

        print("PARAMETRI", flush=True)
        print(f"mae: {'sar_to_opt' if sar_to_opt else 'opt_to_sar'} | n_images: {n_images1 if sar_to_opt else n_images2}", flush=True)
        print(f"data input: {'SAR' if sar_to_opt else 'OPT'} | data target: {'OPT' if sar_to_opt else 'SAR'}", flush=True)
        print(f"channel_in: {n_channels1 if sar_to_opt else n_channels2} | channel_out: {n_channels2 if sar_to_opt else n_channels1}\n", flush=True)

        paired_ids = sorted(set(input_list) & set(target_list))
        dataset_pair = PairsLoader(
            listIDs=paired_ids,
            input_cache_dir=dataset_input.cache_dir,
            target_cache_dir=dataset_target.cache_dir,
            mask=mask,
            mask_precentage=mask_precentage,
            mask_square_size=mask_square_size,
        )

    except Exception as e:
        print(f"EXC -> Eccezione in Loader mae:", {e}, flush=True)

    try:
        dataloader_pair = DataLoader(dataset_pair, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    except Exception as e:
        print(f"EXC -> Eccezione in DataLoader mae {e}", flush=True)

    results_mae = {'lr': [], 'train_loss': []}
    epochs_no_improve = 0

    try:
        for epoch in range(1, epochs + 1):
            model_mae.train()
            total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloader_pair)

            if time_debug:
                t_prev = time.time()

            for im_in, im_target in train_bar:
                if time_debug:
                    torch.cuda.synchronize()
                    t_data = time.time()

                im_in = im_in.to(non_blocking=True, device=device)
                im_target = im_target.to(non_blocking=True, device=device)

                if time_debug:
                    torch.cuda.synchronize()
                    t_transfer = time.time()

                with torch.cuda.amp.autocast():
                    output = model_mae(im_in)
                    loss = loss_fn(output, im_target)

                if time_debug:
                    torch.cuda.synchronize()
                    t_forward = time.time()

                optimizer_mae.zero_grad()
                scaler_mae.scale(loss).backward()
                scaler_mae.step(optimizer_mae)
                scaler_mae.update()

                if time_debug:
                    torch.cuda.synchronize()
                    t_backward = time.time()
                    print(f"dati: {t_data-t_prev:.3f}s | trasferimento: {t_transfer-t_data:.3f}s | "
                        f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                    t_prev = time.time()

                total_num += dataloader_pair.batch_size
                total_loss += loss.item() * dataloader_pair.batch_size
                train_bar.set_description(f'mae Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')

            epoch_loss = total_loss / total_num
            results_mae['lr'].append(optimizer_mae.param_groups[0]['lr'])
            results_mae['train_loss'].append(epoch_loss)

            if epoch_loss < best_loss - min_delta:
                state_dict = model_mae.module.state_dict() if n_gpus > 1 else model_mae.state_dict()
                torch.save(state_dict, f'model_mae_best_{job_name}_{direction}_{dataset}_{epochs}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0
                print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                pd.DataFrame(results_mae).to_csv(f'log_train_mae_{job_name}_{direction}_{dataset}_{epochs}.csv', index_label='epoch')
                try:
                    project_work.log_artifact(name=f"encoder-mae-weights_{job_name}_{direction}_{dataset}_{epochs}", source=f'model_mae_best_{job_name}_{direction}_{dataset}_{epochs}.pth', kind='artifact')
                except Exception as e:    
                    print(f"EXC -> upload pesi mae: {e}", flush=True)
                try:    
                    project_work.log_artifact(name=f"metrics-mae_{job_name}_{direction}_{dataset}_{epochs}", source=f'log_train_mae_{job_name}_{direction}_{dataset}_{epochs}.csv', kind='artifact')
                except Exception as e:
                    print(f"EXC -> upload intermedio mae: {e}", flush=True)
            else:
                epochs_no_improve += 1
                print(f"best loss epoch {epoch} (old): {best_loss}", flush=True)

                if epochs_no_improve >= patience:
                    print(f"Early Stopping mae: epoch {epoch}, best loss: {best_loss}")
                    break

        best_loss_mae = best_loss

    except Exception as e:
        print(f"EXC -> Eccezione in model train mae: {e}", flush=True)

    print(f"OK -> Terminato training mae {direction}, LOSS (MSE): {best_loss_mae}")
       
    return "TERMINATO -> training mae finito"


# TEST

@handler()
def test_encoders_visual(
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
    weights_s1: str = "weights_s1",
    weights_s2: str = "weights_s2",
    n_samples: int = 10,
    recon_channels: list | None = None,
    save_dir: str = "/data/debug_recon",
):
    print("torch version:", torch.__version__, flush=True)
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
 
    torch.backends.cudnn.benchmark = True
 
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")
    print("OK -> Download terminato", flush=True)
 
    os.makedirs(save_dir, exist_ok=True)
 
    try:
        if test_sar:
            s1_dir = os.path.join(dataset_path, "SAR")
            sar_zip_map = {}
            for zip_file in sorted(glob(os.path.join(s1_dir, "SAR_*.zip"))):
                with zipfile.ZipFile(zip_file, 'r') as z:
                    for n in z.namelist():
                        if n.lower().endswith('.tif'):
                            ID = os.path.basename(n).split('_SAR_')[0]
                            sar_zip_map[ID] = zip_file
            train_data_SAR_IDS = list(sar_zip_map.keys())
            random_sar_ids = random.sample(train_data_SAR_IDS, min(n_samples, len(train_data_SAR_IDS))) if train_data_SAR_IDS else []
            print(f"{len(train_data_SAR_IDS)} serie SAR trovate", flush=True)
 
        if test_opt:
            s2_dir = os.path.join(dataset_path, "OPT")
            opt_zip_map = {}
            for zip_file in sorted(glob(os.path.join(s2_dir, "OPT_*.zip"))):
                with zipfile.ZipFile(zip_file, 'r') as z:
                    for n in z.namelist():
                        if n.lower().endswith('.tif'):
                            ID = os.path.basename(n).split('_OPT_')[0]
                            opt_zip_map[ID] = zip_file
            train_data_OPT_IDS = list(opt_zip_map.keys())
            random_opt_ids = random.sample(train_data_OPT_IDS, min(n_samples, len(train_data_OPT_IDS))) if train_data_OPT_IDS else []
            print(f"{len(train_data_OPT_IDS)} serie OPT trovate", flush=True)
 
    except Exception as e:
        print(f"EXC -> Eccezione recupero liste: {e}", flush=True)
 
    if test_sar:
        try:
            s1_path = project_work.get_artifact(weights_s1).download("/data/weights_s1.pth")
        except Exception as e:
            print(f"EXC -> {weights_s1} non trovato: {e}", flush=True)
            s1_path = None
 
        if s1_path:
            modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=output_dim, n_images=n_images1, mamba=mamba).to(device)
 
            state_dict = torch.load(s1_path, map_location=device)
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            modelS1.load_state_dict(state_dict)
            modelS1.eval()
 
            datasetS1 = Singlemodal_Loader(
                listIDs=random_sar_ids, root=dataset_path, zip_map=sar_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images1,
                n_channels=n_channels1, data_type='SAR'
            )
            loaderS1 = torch.utils.data.DataLoader(datasetS1, batch_size=1, shuffle=False)
 
            print("SERIE SAR:", flush=True)
            with torch.no_grad():
                for i, im in enumerate(loaderS1):
                    im = im.to(device)
 
                    try:
                        latent_vector = modelS1.encoder(im)
                    except AttributeError as e:
                        print(f"EXC -> nome encoder s1: {e}", flush=True)
                        break
 
                    vector = latent_vector.cpu().numpy().flatten()
                    print(f"ID {i}: {random_sar_ids[i]} | shape: {list(latent_vector.shape)}", flush=True)
                    print(f"VEC: {np.round(vector, 4)}\n", flush=True)
 
                    save_reconstruction_pngs(
                        modelS1, im, save_dir=save_dir,
                        prefix=f"SAR_{random_sar_ids[i]}",
                        channels=recon_channels, device=device,
                    )
 
            del modelS1
            torch.cuda.empty_cache()
 
    if test_opt:
        try:
            s2_path = project_work.get_artifact(weights_s2).download("/data/weights_s2.pth")
        except Exception as e:
            print(f"EXC -> {weights_s2} non trovato: {e}", flush=True)
            s2_path = None
 
        if s2_path:
            modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=output_dim, n_images=n_images2, mamba=mamba).to(device)
 
            state_dict = torch.load(s2_path, map_location=device)
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            modelS2.load_state_dict(state_dict)
            modelS2.eval()
 
            datasetS2 = Singlemodal_Loader(
                listIDs=random_opt_ids, root=dataset_path, zip_map=opt_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images2,
                n_channels=n_channels2, data_type='OPT'
            )
            loaderS2 = torch.utils.data.DataLoader(datasetS2, batch_size=1, shuffle=False)
 
            print("SERIE OPT:", flush=True)
            with torch.no_grad():
                for i, im in enumerate(loaderS2):
                    im = im.to(device)
 
                    try:
                        latent_vector = modelS2.encoder(im)
                    except AttributeError as e:
                        print(f"EXC -> nome encoder s2: {e}", flush=True)
                        break
 
                    vector = latent_vector.cpu().numpy().flatten()
                    print(f"ID {i}: {random_opt_ids[i]} | shape: {list(latent_vector.shape)}", flush=True)
                    print(f"VEC: {np.round(vector, 4)}\n", flush=True)
 
                    save_reconstruction_pngs(
                        modelS2, im, save_dir=save_dir,
                        prefix=f"OPT_{random_opt_ids[i]}",
                        channels=recon_channels, device=device,
                    )
 
            del modelS2
            torch.cuda.empty_cache()
 
    # zippo tutti i PNG e li carico come artifact, altrimenti spariscono col container del job
    zip_base = save_dir.rstrip("/")
    zip_path = f"{zip_base}.zip"
    try:
        shutil.make_archive(zip_base, 'zip', save_dir)
        project_work.log_artifact(
            name=f"test-encoders-reconstructions_{dataset}",
            kind="artifact",
            source=zip_path
        )
        print(f"OK -> {zip_path} caricato come artifact", flush=True)
    except Exception as e:
        print(f"EXC -> upload zip ricostruzioni: {e}", flush=True)
 
    return "TERMINATO"


import os
import shutil
import zipfile
from glob import glob


def _write_cache_parts(cache_dir, local_root, modality_label, max_part_bytes):
    """
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


@handler()
def crop_norm_artifact(
    dataset: str = "Standard",
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,   # SAR
    n_images2: int = 4, n_channels2: int = 10,  # OPT
    max_part_gb: float = 15.0,  # sotto ai 15.8GB del SAR_N.zip piu' pesante, gia' confermato funzionante
):
    project_data = dh.get_project("datasets")
    max_part_bytes = int(max_part_gb * 1024 ** 3)

    local_root = "/data/floods_cache_upload"
    os.makedirs(local_root, exist_ok=True)

    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")
    print("OK -> Download terminato", flush=True)

    try:
        # ---------------- SAR ----------------
        s1_dir = os.path.join(dataset_path, "SAR")
        sar_zip_map = {}
        for zip_file in sorted(glob(os.path.join(s1_dir, "SAR_*.zip"))):
            with zipfile.ZipFile(zip_file, 'r') as z:
                for n in z.namelist():
                    if n.lower().endswith('.tif'):
                        ID = os.path.basename(n).split('_SAR_')[0]
                        sar_zip_map[ID] = zip_file
        print(f"{len(sar_zip_map)} serie SAR trovate", flush=True)

        # costruisce /data/cache_SAR/{ID}.npy per ogni serie (ritaglio+normalizzazione)
        datasetS1 = Singlemodal_Loader(
            listIDs=list(sar_zip_map.keys()), root=dataset_path, zip_map=sar_zip_map,
            transform=None, patch_size=patch_size, n_images=n_images1,
            n_channels=n_channels1, data_type='SAR'
        )

        for zip_path in set(sar_zip_map.values()):
            if os.path.exists(zip_path):
                os.remove(zip_path)
        print("OK -> ZIP SAR grezzi eliminati", flush=True)

        _write_cache_parts(datasetS1.cache_dir, local_root, "SAR", max_part_bytes)

        # ---------------- OPT ----------------
        s2_dir = os.path.join(dataset_path, "OPT")
        opt_zip_map = {}
        for zip_file in sorted(glob(os.path.join(s2_dir, "OPT_*.zip"))):
            with zipfile.ZipFile(zip_file, 'r') as z:
                for n in z.namelist():
                    if n.lower().endswith('.tif'):
                        ID = os.path.basename(n).split('_OPT_')[0]
                        opt_zip_map[ID] = zip_file
        print(f"{len(opt_zip_map)} serie OPT trovate", flush=True)

        datasetS2 = Singlemodal_Loader(
            listIDs=list(opt_zip_map.keys()), root=dataset_path, zip_map=opt_zip_map,
            transform=None, patch_size=patch_size, n_images=n_images2,
            n_channels=n_channels2, data_type='OPT'
        )

        for zip_path in set(opt_zip_map.values()):
            if os.path.exists(zip_path):
                os.remove(zip_path)
        print("OK -> ZIP OPT grezzi eliminati", flush=True)

        _write_cache_parts(datasetS2.cache_dir, local_root, "OPT", max_part_bytes)

        # ---------------- upload unico ----------------
        print("Upload dell'intera cartella come artifact unico...", flush=True)
        project_data.log_artifact(
            name=f"Floods_{dataset}_crop_norm",
            kind='artifact',
            source=local_root,  # directory locale -> un solo artifact, struttura SAR/ e OPT/ preservata
        )
        print("OK -> artifact unico caricato (sottocartelle SAR/ e OPT/)", flush=True)

        shutil.rmtree(local_root, ignore_errors=True)
        print("OK -> pulizia locale completata", flush=True)

    except Exception as e:
        print(f"EXC -> costruzione cache fallita: {e}", flush=True)

    return "TERMINATO -> cache precalcolata caricata come artifact unico (SAR/, OPT/)"


