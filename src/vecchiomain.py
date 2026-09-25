
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
from moco.builder import MoCo2encoders_2d


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


# notebook -> autoencoders_1D

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
    print('Using device:', device, flush=True)

    n_gpus = torch.cuda.device_count()
    print(f"GPU: {n_gpus}", "\n", flush=True)

    torch.backends.cudnn.benchmark = True
    
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    # download dataset
    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")
    print("OK -> Download terminato\n" )

    try:

        if train_sar:
            # recupero id serie SAR 
            sar_dir = os.path.join(dataset_path, "SAR")
            sar_zip_map = {}
            for zip_file in sorted(glob(os.path.join(sar_dir, "SAR_*.zip"))):
                with zipfile.ZipFile(zip_file, 'r') as z:
                    for n in z.namelist():
                        if n.lower().endswith('.tif'):
                            ID = os.path.basename(n).split('_SAR_')[0]
                            sar_zip_map[ID] = zip_file
            train_data_SAR_IDS = list(sar_zip_map.keys())
            print(f"{len(train_data_SAR_IDS)} serie SAR trovate")

        if train_opt:
            # recupero id serie OPT 
            opt_dir = os.path.join(dataset_path, "OPT")
            opt_zip_map = {}
            for zip_file in sorted(glob(os.path.join(opt_dir, "OPT_*.zip"))):
                with zipfile.ZipFile(zip_file, 'r') as z:
                    for n in z.namelist():
                        if n.lower().endswith('.tif'):
                            ID = os.path.basename(n).split('_OPT_')[0]
                            opt_zip_map[ID] = zip_file
            train_data_OPT_IDS = list(opt_zip_map.keys())
            print(f"{len(train_data_OPT_IDS)} serie OPT trovate")

        print("\n")
    except Exception as e:
        print(f"EXC -> recupero liste: {e}", flush=True)    


    # TRAINING SAR
    if train_sar:
    
        print("--- Training SAR ---\n")
        modelSAR = Singlemodal_CAE(input_dim=n_channels1, output_dim=output_dim, n_images=n_images1, mamba=mamba).to(device)

        if n_gpus > 1:
            modelSAR = nn.DataParallel(modelSAR)
        optimizerSAR = torch.optim.Adam(modelSAR.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss().to(device)
        scalerSAR = torch.cuda.amp.GradScaler()
        best_loss = float('inf')

        resultsSAR = {'lr': [], 'train_loss': []}

        # resume = True, carica pesi e metriche vecchio train
        if resume:
            try:
                w_path = project_work.get_artifact(f"autoencoder1D_SAR_weights_{job_name}").download("/data")
                state_dict = torch.load(w_path, map_location=device)
                (modelSAR.module if n_gpus > 1 else modelSAR).load_state_dict(state_dict)
                print(f"OK -> pesi SAR caricati: {job_name}", flush=True)
            except Exception as e:
                print(f"EXC -> nessun peso SAR trovato: inizializzazione casuale: {e}", flush=True)

            try:
                m_path = project_work.get_artifact(f"autoencoder1D_SAR_metrics_{job_name}").download("/data")
                prev_df = pd.read_csv(m_path)
                best_loss = prev_df['train_loss'].min()
                resultsSAR = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
                print(f"OK -> metriche SAR caricate: best_loss={best_loss}", flush=True)
            except Exception as e:
                print(f"EXC -> nessuna metrica SAR trovata: {e}", flush=True)

            print("\n")    

        try:
            datasetSAR = Singlemodal_Loader(
                listIDs=train_data_SAR_IDS, root=dataset_path, zip_map=sar_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images1,
                n_channels=n_channels1, data_type='SAR'
            )
            for zip_path in set(sar_zip_map.values()):
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            print("OK -> Zip SAR eliminati\n", flush=True)
        except Exception as e:
            print(f"EXC -> Loader SAR:", {e}, "\n", flush=True)

        # check per mse loss
        # calcolo della mse se l'autoencoder producesse sempre la media -> mean_mse_loss
        # train loss buona se inferiose alla mean_mse_loss
        try:
            sample_ids = train_data_SAR_IDS[:100]
            mean_values = None
            n = 0

            for ID in sample_ids:
                im = np.load(os.path.join(datasetSAR.cache_dir, f"{ID}.npy"))
                if mean_values is None:
                    mean_values = np.zeros_like(im, dtype=np.float64)
                mean_values += im
                n += 1
            mean_image = (mean_values / n).astype(np.float32)

            total_squared_error = 0.0
            total_count = 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetSAR.cache_dir, f"{ID}.npy"))
                total_squared_error += np.sum((im - mean_image) ** 2)
                total_values += im.size

            mean_mse_loss = total_squared_error/total_values     

            print(f"mean mse loss SAR: {mean_mse_loss:.6f}", "\n", flush=True)
        except Exception as e:
            print(f"EXC -> mean mse loss SAR: {e}", "\n", flush=True)    

        try:
            dataloaderSAR = DataLoader(datasetSAR, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        except Exception as e:
            print(f"EXC -> DataLoader SAR {e}", flush=True)

        epochs_no_improve = 0

        # libreria time utilizzata per debug, tempo training
        try:
            for epoch in range(1, epochs + 1):
                modelSAR.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderSAR, mininterval=5.0)

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
                        output = modelSAR(im)
                        loss = loss_fn(output, im)

                    if time_debug:
                        torch.cuda.synchronize()
                        t_forward = time.time()

                    optimizerSAR.zero_grad()
                    scalerSAR.scale(loss).backward()
                    scalerSAR.step(optimizerSAR)
                    scalerSAR.update()

                    if time_debug:
                        torch.cuda.synchronize()
                        t_backward = time.time()
                        print(f"dati: {t_data-t_prev:.3f}s | transfer: {t_transfer-t_data:.3f}s | "
                            f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                        t_prev = time.time()

                    total_num += dataloaderSAR.batch_size
                    total_loss += loss.item() * dataloaderSAR.batch_size
                    train_bar.set_description(f'SAR Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')

                epoch_loss = total_loss / total_num
                resultsSAR['lr'].append(optimizerSAR.param_groups[0]['lr'])
                resultsSAR['train_loss'].append(epoch_loss)

                # loss migliorata -> salva metriche e pesi
                pd.DataFrame(resultsSAR).to_csv(f'autoencoder1D_SAR_log_{job_name}.csv', index_label='epoch')

                # refresh token e progetto 
                try:
                    dh.refresh_token()
                    print("OK -> refresh token", flush=True)
                except Exception as e:
                    print(f"EXC -> refresh token: {e}", flush=True)

                try:
                    project_work = dh.get_project("floods")
                    print("OK -> get project\n", flush=True)
                except Exception as e:
                    print(f"EXC -> get project: {e}", "\n", flush=True)

                try:    
                    project_work.log_artifact(name=f"autoencoder1D_SAR_metrics_{job_name}", source=f'autoencoder1D_SAR_log_{job_name}.csv', kind='artifact')
                    print("OK -> metriche SAR salvate", flush=True)
                except Exception as e:
                    print(f"EXC -> salvataggio metriche SAR: {e}", flush=True)                

                if epoch_loss < best_loss - min_delta:
                    state_dict = modelSAR.module.state_dict() if n_gpus > 1 else modelSAR.state_dict()
                    torch.save(state_dict, f'autoencoder1D_SAR_model_best_{job_name}.pth')
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                    try:
                        project_work.log_artifact(name=f"autoencoder1D_SAR_weights_{job_name}", source=f'autoencoder1D_SAR_model_best_{job_name}.pth', kind='artifact')
                        print("OK -> pesi SAR salvati\n", flush=True)
                    except Exception as e:
                        print(f"EXC -> salvataggio pesi SAR: {e}", "\n", flush=True)

                else:
                    epochs_no_improve += 1
                    print(f"loss epoch {epoch}: {epoch_loss}", flush=True)
                    print(f"best loss epoch {epoch} (old): {best_loss}", "\n", flush=True)

                    if epochs_no_improve >= patience:
                        print(f"Early Stopping epoch {epoch} -> best loss: {best_loss}", "\n", flush=True)
                        break

        except Exception as e:
            print(f"EXC -> training SAR: {e}", "\n", flush=True)

        print(f"OK -> Terminato training SAR, LOSS (MSE): {best_loss}", "\n", flush=True)     

        del modelSAR
        torch.cuda.empty_cache() 

        # se train sia sar che opt memoria insufficiente
        # rimozione cartella sar dopo termine train sar
        try:
            cache_sar_dir = "/data/cache_SAR" 
            if os.path.exists(cache_sar_dir):
                print(f"Pulizia cache_SAR: {cache_sar_dir}", flush=True)
                shutil.rmtree(cache_sar_dir)
                print("OK -> cache_SAR rimossa\n", flush=True)
            else:
                print("cache_SAR non trovata\n", flush=True)

        except Exception as e:
            print(f"EXC -> pulizia cache_SAR: {e}", "\n", flush=True)


    # TRAINING OTTICO
    # stessa logica sar
    if train_opt:
    
        print("--- Training OPT ---\n")
        modelOPT = Singlemodal_CAE(input_dim=n_channels2, output_dim=output_dim, n_images=n_images2, mamba=mamba).to(device)
        if n_gpus > 1:
            modelOPT = nn.DataParallel(modelOPT)
        optimizerOPT = torch.optim.Adam(modelOPT.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss().to(device)
        scalerOPT = torch.cuda.amp.GradScaler()
        best_loss = float('inf')

        resultsOPT = {'lr': [], 'train_loss': []}

        if resume:
            try:
                w_path = project_work.get_artifact(f"autoencoder1D_OPT_weights_{job_name}").download("/data")
                state_dict = torch.load(w_path, map_location=device)
                (modelOPT.module if n_gpus > 1 else modelOPT).load_state_dict(state_dict)
                print(f"OK -> pesi OPT caricati: {job_name}", flush=True)
            except Exception as e:
                print(f"EXC -> nessun peso OPT trovato: inizializzazione casuale: {e}", flush=True)

            try:
                m_path = project_work.get_artifact(f"autoencoder1D_OPT_metrics_{job_name}").download("/data")
                prev_df = pd.read_csv(m_path)
                best_loss = prev_df['train_loss'].min()
                resultsOPT = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
                print(f"OK -> metriche OPT caricate -> best_loss={best_loss}", flush=True)
            except Exception as e:
                print(f"EXC -> nessuna metrica OPT trovata: {e}", flush=True)

            print("\n")    

        try:
            datasetOPT = Singlemodal_Loader(
                listIDs=train_data_OPT_IDS, root=dataset_path, zip_map=opt_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images2,
                n_channels=n_channels2, data_type='OPT'
            )
            for zip_path in set(opt_zip_map.values()):
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            print("OK -> Zip OPT eliminati\n", flush=True)
        except Exception as e:
            print(f"EXC -> Loader OPT:", {e}, "\n", flush=True)

        try:
            sample_ids = train_data_OPT_IDS[:100]
            mean_values = None
            n = 0

            for ID in sample_ids:
                im = np.load(os.path.join(datasetOPT.cache_dir, f"{ID}.npy"))
                if mean_values is None:
                    mean_values = np.zeros_like(im, dtype=np.float64)
                mean_values += im
                n += 1
            mean_image = (mean_values / n).astype(np.float32)

            total_squared_error = 0.0
            total_values = 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetOPT.cache_dir, f"{ID}.npy"))
                total_squared_error += np.sum((im - mean_image) ** 2)
                total_values += im.size

            mean_mse_loss = total_squared_error/total_values     

            print(f"mean mse loss OPT: {mean_mse_loss:.6f}", "\n", flush=True)
        except Exception as e:
            print(f"EXC -> mean mse loss OPT: {e}", "\n", flush=True) 

        try:
            dataloaderOPT = DataLoader(datasetOPT, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        except Exception as e:
            print(f"EXC -> DataLoader OPT {e}", flush=True)

        epochs_no_improve = 0

        try:
            for epoch in range(1, epochs + 1):
                modelOPT.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderOPT)

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
                        output = modelOPT(im)
                        loss = loss_fn(output, im)

                    if time_debug:
                        torch.cuda.synchronize()
                        t_forward = time.time()

                    optimizerOPT.zero_grad()
                    scalerOPT.scale(loss).backward()
                    scalerOPT.step(optimizerOPT)
                    scalerOPT.update()

                    if time_debug:
                        torch.cuda.synchronize()
                        t_backward = time.time()
                        print(f"data: {t_data-t_prev:.3f}s | transfer: {t_transfer-t_data:.3f}s | "
                            f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                        t_prev = time.time()

                    total_num += dataloaderOPT.batch_size
                    total_loss += loss.item() * dataloaderOPT.batch_size
                    train_bar.set_description(f'OPT Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')

                epoch_loss = total_loss / total_num
                resultsOPT['lr'].append(optimizerOPT.param_groups[0]['lr'])
                resultsOPT['train_loss'].append(epoch_loss)

                pd.DataFrame(resultsOPT).to_csv(f'autoencoder1D_OPT_log_{job_name}.csv', index_label='epoch')

                try:
                    dh.refresh_token()
                    print("OK -> refresh token", flush=True)
                except Exception as e:
                    print(f"EXC -> refresh token: {e}", flush=True)

                try:
                    project_work = dh.get_project("floods")
                    print("OK -> get project\n", flush=True)
                except Exception as e:
                    print(f"EXC -> get project: {e}", "\n", flush=True)

                try:
                    project_work.log_artifact(name=f"autoencoder1D_OPT_metrics_{job_name}", source=f'autoencoder1D_OPT_log_{job_name}.csv', kind='artifact')
                    print("OK -> metriche OPT salvate", flush=True)
                except Exception as e:
                    print(f"EXC -> salvataggio metriche OPT: {e}", flush=True)                

                if epoch_loss < best_loss - min_delta:
                    state_dict = modelOPT.module.state_dict() if n_gpus > 1 else modelOPT.state_dict()
                    torch.save(state_dict, f'autoencoder1D_OPT_model_best_{job_name}.pth')
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                    try:
                        project_work.log_artifact(name=f"autoencoder1D_OPT_weights_{job_name}", source=f'autoencoder1D_OPT_model_best_{job_name}.pth', kind='artifact')
                        print("OK -> pesi OPT salvati\n", flush=True)
                    except Exception as e:
                        print(f"EXC -> salvataggio pesi OPT: {e}", "\n", flush=True)

                else:
                    epochs_no_improve += 1
                    print(f"loss epoch {epoch}: {epoch_loss}", flush=True)
                    print(f"best loss epoch {epoch} (old): {best_loss}", "\n", flush=True)

                    if epochs_no_improve >= patience:
                        print(f"Early Stopping OPT: epoch {epoch}, best loss: {best_loss}", "\n", flush=True)
                        break

        except Exception as e:
            print(f"EXC -> training OPT: {e}", "\n", flush=True)

        del modelOPT
        torch.cuda.empty_cache()

        print(f"OK -> Terminato training OPT, LOSS (MSE): {best_loss}", "\n", flush=True) 
    
    return "TERMINATO -> training autoencoders1D"


# notebook -> autoencoders_2D

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
    print('Using device:', device, flush=True)
 
    n_gpus = torch.cuda.device_count()
    print(f"GPU: {n_gpus}", "\n", flush=True)
 
    torch.backends.cudnn.benchmark = True
 
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    sar_zip_map = {}
    opt_zip_map = {}
    train_data_SAR_IDS = []
    train_data_OPT_IDS = []
   
    # download dataset
    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}_crop_norm").download("/data/dataset_floods")
    print("OK -> Download terminato\n")


    if train_sar:
        try:
            os.makedirs("/data/cache_SAR", exist_ok=True)
            part_files = sorted(glob(os.path.join(dataset_path, "SAR", "part*.zip")))
            if part_files:
                for part_path in part_files:
                    with zipfile.ZipFile(part_path, 'r') as z:
                        z.extractall("/data/cache_SAR")
                train_data_SAR_IDS = [f[:-4] for f in os.listdir("/data/cache_SAR") if f.endswith('.npy')]
                print(f"OK -> caricate {len(train_data_SAR_IDS)} serie SAR", flush=True)
        except Exception as e:
            print(f"EXC -> caricamento serie SAR: {e}", flush=True)

    if train_opt:
        try:
            os.makedirs("/data/cache_OPT", exist_ok=True)
            part_files = sorted(glob(os.path.join(dataset_path, "OPT", "part*.zip")))
            if part_files:
                for part_path in part_files:
                    with zipfile.ZipFile(part_path, 'r') as z:
                        z.extractall("/data/cache_OPT")
                train_data_OPT_IDS = [f[:-4] for f in os.listdir("/data/cache_OPT") if f.endswith('.npy')]
                print(f"OK -> caricate {len(train_data_OPT_IDS)} serie OPT", flush=True)
        except Exception as e:
            print(f"EXC -> caricamento serie OPT: {e}", flush=True)

    print("\n")        
 
    
    # TRAINING SAR
    if train_sar:
 
        print("--- Training SAR ---\n")
        modelSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=output_dim, n_images=n_images1, n_head=8, d_k=8, ltae=ltae).to(device)
        
        if n_gpus > 1:
            modelSAR = nn.DataParallel(modelSAR)
        optimizerSAR = torch.optim.Adam(modelSAR.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss().to(device)
        scalerSAR = torch.cuda.amp.GradScaler()
        best_loss = float('inf')
 
        resultsSAR = {'lr': [], 'train_loss': []}
 
        # resume = True, carica pesi e metriche vecchio train
        if resume:
            try:
                w_path = project_work.get_artifact(f"autoencoder2D_SAR_weights_{job_name}").download("/data")
                state_dict = torch.load(w_path, map_location=device)
                (modelSAR.module if n_gpus > 1 else modelSAR).load_state_dict(state_dict)
                print(f"OK -> pesi SAR caricati -> {job_name}", flush=True)
            except Exception as e:
                print(f"EXC -> nessun peso SAR trovato: inizializzazione casuale: {e}", flush=True)
 
            try:
                m_path = project_work.get_artifact(f"autoencoder2D_SAR_metrics_{job_name}").download("/data")
                prev_df = pd.read_csv(m_path)
                best_loss = prev_df['train_loss'].min()
                resultsSAR = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
                print(f"OK -> metriche SAR caricate:  best_loss={best_loss}", flush=True)
            except Exception as e:
                print(f"EXC -> nessuna metrica SAR trovata: {e}", flush=True)

            print("\n")    
 
        try:
            datasetSAR = Singlemodal_Loader(
                listIDs=train_data_SAR_IDS, root=dataset_path, zip_map=sar_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images1,
                n_channels=n_channels1, data_type='SAR'
            )
            for zip_path in set(sar_zip_map.values()):
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            print("OK -> Zip SAR eliminati\n", flush=True)
        except Exception as e:
            print(f"EXC -> Loader SAR:", {e}, "\n", flush=True)
 
        # check per mse loss
        # calcolo della mse se l'autoencoder producesse sempre la media -> mean_mse_loss
        # train loss buona se inferiose alla mean_mse_loss
        try:
            sample_ids = train_data_SAR_IDS[:100]
            mean_values = None
            n = 0

            for ID in sample_ids:
                im = np.load(os.path.join(datasetSAR.cache_dir, f"{ID}.npy"))
                if mean_values is None:
                    mean_values = np.zeros_like(im, dtype=np.float64)
                mean_values += im
                n += 1
            mean_image = (mean_values / n).astype(np.float32)

            total_squared_error = 0.0
            total_count = 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetSAR.cache_dir, f"{ID}.npy"))
                total_squared_error += np.sum((im - mean_image) ** 2)
                total_values += im.size

            mean_mse_loss = total_squared_error/total_values     

            print(f"mean mse loss SAR: {mean_mse_loss:.6f}", "\n", flush=True)
        except Exception as e:
            print(f"EXC -> mean mse loss SAR: {e}", "\n", flush=True)

 
        try:
            dataloaderSAR = DataLoader(datasetSAR, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        except Exception as e:
            print(f"EXC -> DataLoader SAR {e}", flush=True)
 
        epochs_no_improve = 0

        # libreria time utilizzata per debug, tempo training
        try:
            for epoch in range(1, epochs + 1):
                modelSAR.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderSAR, mininterval=5.0)
 
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
                        output = modelSAR(im)
                        loss = loss_fn(output, im)
 
                    if time_debug:
                        torch.cuda.synchronize()
                        t_forward = time.time()
 
                    optimizerSAR.zero_grad()
                    scalerSAR.scale(loss).backward()
                    scalerSAR.step(optimizerSAR)
                    scalerSAR.update()
 
                    if time_debug:
                        torch.cuda.synchronize()
                        t_backward = time.time()
                        print(f"data: {t_data-t_prev:.3f}s | transfer: {t_transfer-t_data:.3f}s | "
                            f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                        t_prev = time.time()
 
                    total_num += dataloaderSAR.batch_size
                    total_loss += loss.item() * dataloaderSAR.batch_size
                    train_bar.set_description(f'SAR Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')
 
                epoch_loss = total_loss / total_num
                resultsSAR['lr'].append(optimizerSAR.param_groups[0]['lr'])
                resultsSAR['train_loss'].append(epoch_loss)
 
                # loss migliorata -> salva metriche e pesi
                pd.DataFrame(resultsSAR).to_csv(f'autoencoder1D_SAR_log_{job_name}.csv', index_label='epoch')

                # evitare scadenza token auth 
                try:
                    dh.refresh_token()
                    print("OK -> refresh token", flush=True)
                except Exception as e:
                    print(f"EXC -> refresh token: {e}", flush=True)

                try:
                    project_work = dh.get_project("floods")
                    print("OK -> get project\n", flush=True)
                except Exception as e:
                    print(f"EXC -> get project: {e}", "\n", flush=True)

                try:    
                    project_work.log_artifact(name=f"autoencoder1D_SAR_metrics_{job_name}", source=f'autoencoder1D_SAR_log_{job_name}.csv', kind='artifact')
                    print("OK -> metriche SAR salvate", flush=True)
                except Exception as e:
                    print(f"EXC -> salvataggio metriche SAR: {e}", flush=True)                

                if epoch_loss < best_loss - min_delta:
                    state_dict = modelSAR.module.state_dict() if n_gpus > 1 else modelSAR.state_dict()
                    torch.save(state_dict, f'autoencoder1D_SAR_model_best_{job_name}.pth')
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                    try:
                        project_work.log_artifact(name=f"autoencoder1D_SAR_weights_{job_name}", source=f'autoencoder1D_SAR_model_best_{job_name}.pth', kind='artifact')
                        print("OK -> pesi SAR salvati\n", flush=True)
                    except Exception as e:
                        print(f"EXC -> salvataggio pesi SAR: {e}", "\n", flush=True)

                else:
                    epochs_no_improve += 1
                    print(f"loss epoch {epoch}: {epoch_loss}", flush=True)
                    print(f"best loss epoch {epoch} (old): {best_loss}", "\n", flush=True)

                    if epochs_no_improve >= patience:
                        print(f"Early Stopping epoch {epoch} -> best loss: {best_loss}", "\n", flush=True)
                        break

 
        except Exception as e:
            print(f"EXC -> training SAR: {e}", "\n", flush=True)

        print(f"OK -> Terminato training SAR, LOSS (MSE): {best_loss}",  "\n", flush=True) 
 
        del modelSAR
        torch.cuda.empty_cache()

        # se train sia sar che opt memoria insufficiente
        # rimozione cartella sar dopo termine train sar 
        try:
            cache_sar_dir = "/data/cache_SAR" 
            if os.path.exists(cache_sar_dir):
                print(f"Pulizia cache_SAR: {cache_sar_dir}", flush=True)
                shutil.rmtree(cache_sar_dir)
                print("OK -> cache_SAR rimossa\n", flush=True)
            else:
                print("cache_SAR non trovata\n", flush=True)

        except Exception as e:
            print(f"EXC -> pulizia cartella SAR: {e}", "\n", flush=True)

 
    # TRAINING OTTICO
    # stessa logica sar
    if train_opt:
 
        print("\n --- Training OPT ---\n")
        modelOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=output_dim, n_images=n_images2, n_head=8, d_k=8, ltae=ltae).to(device)
        if n_gpus > 1:
            modelOPT = nn.DataParallel(modelOPT)
        optimizerOPT = torch.optim.Adam(modelOPT.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss().to(device)
        scalerOPT = torch.cuda.amp.GradScaler()
        best_loss = float('inf')
 
        resultsOPT = {'lr': [], 'train_loss': []}
 
        if resume:
            try:
                w_path = project_work.get_artifact(f"autoencoder2D_OPT_weights_{job_name}").download("/data")
                state_dict = torch.load(w_path, map_location=device)
                (modelOPT.module if n_gpus > 1 else modelOPT).load_state_dict(state_dict)
                print(f"OK -> pesi OPT caricati: {job_name}", flush=True)
            except Exception as e:
                print(f"EXC -> nessun peso OPT trovato, inizializzazione casuale: {e}", flush=True)

            try:
                m_path = project_work.get_artifact(f"autoencoder2D_OPT_metrics_{job_name}").download("/data")
                prev_df = pd.read_csv(m_path)
                best_loss = prev_df['train_loss'].min()
                resultsOPT = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
                print(f"OK -> metriche OPT caricate: best_loss={best_loss}", flush=True)
            except Exception as e:
                print(f"EXC -> nessuna metrica OPT trovata: {e}", flush=True)

            print("\n")    
 
        try:
            datasetOPT = Singlemodal_Loader(
                listIDs=train_data_OPT_IDS, root=dataset_path, zip_map=opt_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images2,
                n_channels=n_channels2, data_type='OPT'
            )
            for zip_path in set(opt_zip_map.values()):
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            print("OK -> Zip OPT eliminati\n", flush=True)
        except Exception as e:
            print(f"EXC -> Loader OPT:", {e}, "\n", flush=True)

        try:
            sample_ids = train_data_OPT_IDS[:100]
            mean_values = None
            n = 0

            for ID in sample_ids:
                im = np.load(os.path.join(datasetOPT.cache_dir, f"{ID}.npy"))
                if mean_values is None:
                    mean_values = np.zeros_like(im, dtype=np.float64)
                mean_values += im
                n += 1
            mean_image = (mean_values / n).astype(np.float32)

            total_squared_error = 0.0
            total_values = 0
            for ID in sample_ids:
                im = np.load(os.path.join(datasetOPT.cache_dir, f"{ID}.npy"))
                total_squared_error += np.sum((im - mean_image) ** 2)
                total_values += im.size

            mean_mse_loss = total_squared_error/total_values     

            print(f"mean mse loss OPT: {mean_mse_loss:.6f}", "\n", flush=True)
        except Exception as e:
            print(f"EXC -> mean mse loss OPT: {e}", "\n",  flush=True) 

        try:
            dataloaderOPT = DataLoader(datasetOPT, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        except Exception as e:
            print(f"EXC -> DataLoader OPT {e}", flush=True)
 
        epochs_no_improve = 0
 
        try:
            for epoch in range(1, epochs + 1):
                modelOPT.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderOPT)
 
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
                        output = modelOPT(im)
                        loss = loss_fn(output, im)
 
                    if time_debug:
                        torch.cuda.synchronize()
                        t_forward = time.time()
 
                    optimizerOPT.zero_grad()
                    scalerOPT.scale(loss).backward()
                    scalerOPT.step(optimizerOPT)
                    scalerOPT.update()
 
                    if time_debug:
                        torch.cuda.synchronize()
                        t_backward = time.time()
                        print(f"data: {t_data-t_prev:.3f}s | transfer: {t_transfer-t_data:.3f}s | "
                            f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                        t_prev = time.time()
 
                    total_num += dataloaderOPT.batch_size
                    total_loss += loss.item() * dataloaderOPT.batch_size
                    train_bar.set_description(f'OPT Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')
 
                epoch_loss = total_loss / total_num
                resultsOPT['lr'].append(optimizerOPT.param_groups[0]['lr'])
                resultsOPT['train_loss'].append(epoch_loss)
 
                pd.DataFrame(resultsOPT).to_csv(f'autoencoder2D_OPT_log_{job_name}.csv', index_label='epoch')
                
                try:
                    dh.refresh_token()
                    print("OK -> refresh token", flush=True)
                except Exception as e:
                    print(f"EXC -> refresh token: {e}", flush=True)

                try:
                    project_work = dh.get_project("floods")
                    print("OK -> get project\n", flush=True)
                except Exception as e:
                    print(f"EXC -> get project: {e}", "\n", flush=True)

                try:
                    project_work.log_artifact(name=f"autoencoder2D_OPT_metrics_{job_name}", source=f'autoencoder2D_OPT_log_{job_name}.csv', kind='artifact')
                    print("OK -> metriche OPT salvate", flush=True)
                except Exception as e:
                    print(f"EXC -> salvataggio metriche OPT: {e}", flush=True) 
 
                if epoch_loss < best_loss - min_delta:
                    state_dict = modelOPT.module.state_dict() if n_gpus > 1 else modelOPT.state_dict()
                    torch.save(state_dict, f'autoencoder2D_OPT_model_best_{job_name}.pth')
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)
 
                    try:
                        project_work.log_artifact(name=f"autoencoder2D_OPT_weights_{job_name}", source=f'autoencoder2D_OPT_model_best_{job_name}.pth', kind='artifact')
                        print("OK -> pesi OPT salvati\n", flush=True)
                    except Exception as e:
                        print(f"EXC -> salvataggio pesi OPT: {e}", "\n", flush=True)
 
                else:
                    epochs_no_improve += 1
                    print(f"loss epoch {epoch}: {epoch_loss}", flush=True)
                    print(f"best loss epoch {epoch} (old): {best_loss}", flush=True)
 
                    if epochs_no_improve >= patience:
                        print(f"Early Stopping OPT: epoch {epoch}, best loss: {best_loss}", "\n", flush=True)
                        break
 
        except Exception as e:
            print(f"EXC -> training OPT: {e}", "\n", flush=True)

        del modelS2
        torch.cuda.empty_cache()

        print(f"OK -> Terminato training OPT, LOSS (MSE): {best_loss}", flush=True) 
    
    return "TERMINATO -> Training autoencoders2D"



# notebook -> moco_1D

handler()
def train_moco_1D(
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
    resume: bool = False,
    min_delta: float = 1e-4,
    calib_batch_size: int = 32,
    time_debug: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    # download dataset
    print(f"Download dataset: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}_crop_norm").download("/data/dataset_floods")
    print("OK -> Download terminato")

    try:
        sar_dir = os.path.join(dataset_path, "SAR")
        sar_zip_map = {}
        for zip_file in sorted(glob(os.path.join(sar_dir, "SAR_*.zip"))):
            with zipfile.ZipFile(zip_file, 'r') as z:
                for n in z.namelist():
                    if n.lower().endswith('.tif'):
                        ID = os.path.basename(n).split('_SAR_')[0]
                        sar_zip_map[ID] = zip_file
        print(f"{len(sar_zip_map)} serie SAR trovate")

        opt_dir = os.path.join(dataset_path, "OPT")
        opt_zip_map = {}
        for zip_file in sorted(glob(os.path.join(opt_dir, "OPT_*.zip"))):
            with zipfile.ZipFile(zip_file, 'r') as z:
                for n in z.namelist():
                    if n.lower().endswith('.tif'):
                        ID = os.path.basename(n).split('_OPT_')[0]
                        opt_zip_map[ID] = zip_file
        print(f"{len(opt_zip_map)} serie OPT trovate")

        # ordinamento
        train_data_IDS = sorted(set(sar_zip_map) & set(opt_zip_map))
        print(f"{len(train_data_IDS)} serie complete SAR e OPT", flush=True)

        # split 80/20
        random.seed(42)
        shuffled_ids = train_data_IDS.copy()
        random.shuffle(shuffled_ids)

        split_idx = int(len(shuffled_ids) * 0.8)
        train_ids = shuffled_ids[:split_idx]
        test_ids = shuffled_ids[split_idx:]

        print(f"serie train 80%: {len(train_ids)} ", flush=True)
        print(f"serie test 20%:{len(test_ids)} ", "\n", flush=True)

    except Exception as e:
        print(f"EXC -> recupero liste: {e}", flush=True)


    # salvataggio log liste train e test
    with open(f'train_ids_{job_name}.json', 'w') as f:
        json.dump(train_ids, f)
    try:
        project_work.log_artifact(name=f"moco1D_train-ids_{job_name}", source=f'train_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload train_ids: {e}", flush=True)

    with open(f'test_ids_{job_name}.json', 'w') as f:
        json.dump(test_ids, f)
    try:
        project_work.log_artifact(name=f"moco1D_test-ids_{job_name}", source=f'test_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload test: {e}", flush=True)

    # caricamento pesi encoders
    try:
        path_sar = project_work.get_artifact(f"autoencoder1D_SAR_weights_{weights_encoder_sar}").download(f"autoencoder1D_SAR_model_best_{weights_encoder_sar}.pth")
        path_opt = project_work.get_artifact(f"autoencoder1D_OPT_weights_{weights_encoder_opt}").download(f"autoencoder1D_OPT_model_best_{weights_encoder_opt}.pth")

        modelSAR = Singlemodal_CAE(input_dim=n_channels1, output_dim=512, n_images=n_images1, mamba=mamba).to(device)
        modelSAR.load_state_dict(torch.load(path_sar, map_location=device))

        modelOPT = Singlemodal_CAE(input_dim=n_channels2, output_dim=512, n_images=n_images2, mamba=mamba).to(device)
        modelOPT.load_state_dict(torch.load(path_opt, map_location=device))
        print("OK -> Pesi encoders caricati\n", flush=True)
    except Exception as e:
        print(f"EXC -> upload pesi encoders: {e}", flush=True)   


    # TRAINING MOCO
    
    print("--- Training MoCo 1D ---")
    model = MoCo2encoders(
        base_encoder_q=modelOPT.encoder,
        base_encoder_k=modelSAR.encoder,
        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
        symmetric=symmetric, device=device,
    ).to(device)

    results = {'lr': [], 'train_loss': []}
    best_loss = float('inf')
    epochs_no_improve = 0

    # variabili per caricamento pesi
    start_epoch = 1
    queue_restored = False

    # resume = True -> carico pesi vecchi di moco e coda già inzializzata
    if resume:
        try:
            w_path = project_work.get_artifact(f"moco1D_weights_{job_name}").download("/data")
            state_dict = torch.load(w_path, map_location=device)
            model.load_state_dict(state_dict)
            queue_restored = True
            print("OK -> pesi MoCo caricati", flush=True)
        except Exception as e:
            print(f"EXC -> no pesi MoCo vecchi, inizializzazione casuale: {e}", flush=True)
 
        try:
            m_path = project_work.get_artifact(f"moco1D_metrics_{job_name}").download("/data")
            prev_df = pd.read_csv(m_path)
            best_loss = prev_df['train_loss'].min()
            results = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}

            # se carico pesi conteggio riparte da ultima epoca
            start_epoch = len(prev_df) + 1
            print(f"OK -> metriche MoCo caricate, best_loss={best_loss}", flush=True)
        except Exception as e:
            print(f"EXC -> nessuna metrica MoCo trovata: {e}", flush=True)    

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    
    # prima epoca loss molto bassa perche coda inizializzata casualmente
    # lr parte da 0 e poi si stabilizza al valore fissato
    # con batch size = 16 e coda (moco_k) = 4096, la coda si riempie con 256 batch
    # una epoca contiene 1243 batch

    epoch_lenght = None
    base_lr = lr
    batch_step = 0
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
        print(f"EXC -> MoCo2encodersLoader: {e}", flush=True)

    try:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True,
        )
        # quanti batch stanno in un epoca, per Standard e barch size 16 = 1243
        epoch_lenght = len(train_loader) if not queue_restored else None         
    except Exception as e:
        print(f"EXC -> DataLoader: {e}", flush=True)

    # libreria time utilizzata per debug, tempo training
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            model.encoder_q.eval()
            model.encoder_k.eval()

            total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)

            if time_debug:
                t_prev = time.time()

            for im_q, im_k in train_bar:

                # lr sale da 0 a lr incrementalmente ad ogni batch della prima epoca
                # quindi da 0 a 1243, da 1244 lr diventa il paramentro passato nel notebook

                if epoch_lenght is not None and batch_step < epoch_lenght:
                    batch_lr = base_lr * (batch_step + 1) / epoch_lenght
                    for pg in optimizer.param_groups:
                        pg['lr'] = batch_lr
                batch_step += 1     

                if time_debug:
                    torch.cuda.synchronize()
                    t_data = time.time()

                im_q = im_q.to(non_blocking=True, device=device)
                im_k = im_k.to(non_blocking=True, device=device)               
                    
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
                    print(f"data: {t_data-t_prev:.3f}s | transfer: {t_transfer-t_data:.3f}s | "
                        f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                    t_prev = time.time()

                total_num += batch_size
                total_loss += loss.item() * batch_size
                train_bar.set_description(f'MoCo Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')

            epoch_loss = total_loss / total_num
            results['lr'].append(optimizer.param_groups[0]['lr'])
            results['train_loss'].append(epoch_loss)

            # loss migliorata -> salva metriche e pesi
            pd.DataFrame(results).to_csv(f'moco1D_log_{job_name}.csv', index_label='epoch')

            # refresh token e progetto
            try:
                dh.refresh_token()
                print("OK -> refresh token", flush=True)
            except Exception as e:
                print(f"EXC -> refresh token: {e}", flush=True)

            try:
                project_work = dh.get_project("floods")
                print("OK -> get project\n", flush=True)
            except Exception as e:
                print(f"EXC -> get project: {e}", "\n", flush=True)            

            try:
                project_work.log_artifact(name=f"moco1D_metrics_{job_name}", source=f'moco1D_log_{job_name}.csv', kind='artifact')
                print(f"OK -> metriche MoCo salvate", flush=True)
            except Exception as e:
                print(f"EXC -> upload metriche MoCo: {e}", flush=True)

            # skip first epoch for patience
            skip_patience_epoch = (epoch == start_epoch) and (not queue_restored)
 
            if skip_patience_epoch:
                torch.save(model.state_dict(), f'moco1D_model_best_{job_name}.pth')
                print(f"skip patience epoch {epoch}", "\n", flush=True)   

                try:
                    project_work.log_artifact(name=f"moco1D_weights_{job_name}", source=f'moco1D_model_best_{job_name}', kind='artifact')
                    print(f"OK -> pesi salvati", flush=True)
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", flush=True)                             

            elif epoch_loss < best_loss - min_delta:
                torch.save(model.state_dict(), f'moco1D_model_best_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0
                print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                try:
                    project_work.log_artifact(name=f"moco1D_weights_{job_name}", source=f'moco1D_model_best_{job_name}', kind='artifact')
                    print(f"OK -> pesi salvati", flush=True)
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", flush=True) 

            else:
                epochs_no_improve += 1
                print(f"loss epoch {epoch}: {epoch_loss}", flush=True)
                print(f"best loss epoch {epoch} (old): {best_loss}", "\n", flush=True)

                if epochs_no_improve >= patience:
                    print(f"Early Stopping epoch {epoch} -> best loss: {best_loss}", "\n", flush=True)
                    break                    
                
    except Exception as e:
        print(f"EXC -> training MoCo: {e}", "\n", flush=True)

    print(f"OK -> Terminato training MoCo, LOSS (InfoNCE): {best_loss}", "\n", flush=True)
    
    return "TERMINATO -> training moco1D"


@handler()
def train_moco_2D(
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
    workers: int = 0,
    job_name: str = "nome_job",
    dataset: str = "Test",
    weights_encoder_sar: str = "train_sar_1D_v1_Standard_200",
    weights_encoder_opt: str = "train_opt_1D_v1_Standard_200",
    patience: int = 20,
    hidden_channels_dim: int = 64,
    simple_proj: bool = True,
    attn: bool = False,
    min_delta: float = 1e-4,
    resume: bool = True,  
    time_debug: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
  
    torch.backends.cudnn.benchmark = True

    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    sar_zip_map = {}
    opt_zip_map = {}
    train_data_SAR_IDS = []
    train_data_OPT_IDS = []
 
    # download dataset
    print(f"Download dataset: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}_crop_norm").download("/data/dataset_floods")
    print("OK -> Download terminato")
 
    try:
        os.makedirs("/data/cache_SAR", exist_ok=True)
        part_files = sorted(glob(os.path.join(dataset_path, "SAR", "part*.zip")))
        if part_files:
            for part_path in part_files:
                with zipfile.ZipFile(part_path, 'r') as z:
                    z.extractall("/data/cache_SAR")
            train_data_SAR_IDS = [f[:-4] for f in os.listdir("/data/cache_SAR") if f.endswith('.npy')]
            print(f"OK -> SAR caricato: {len(train_data_SAR_IDS)} serie", flush=True)
    except Exception as e:
        print(f"EXC -> caricamento SAR: {e}", flush=True)
 
 
    try:
        os.makedirs("/data/cache_OPT", exist_ok=True)
        part_files = sorted(glob(os.path.join(dataset_path, "OPT", "part*.zip")))
        if part_files:
            for part_path in part_files:
                with zipfile.ZipFile(part_path, 'r') as z:
                    z.extractall("/data/cache_OPT")
            train_data_OPT_IDS = [f[:-4] for f in os.listdir("/data/cache_OPT") if f.endswith('.npy')]
            print(f"OK -> OPT caricato: {len(train_data_OPT_IDS)} serie", flush=True)
    except Exception as e:
        print(f"EXC -> caricamento OPT: {e}", flush=True)
 
    # ordinamento
    train_data_IDS = sorted(set(train_data_SAR_IDS) & set(train_data_OPT_IDS))
    print(f"{len(train_data_IDS)} serie complete SAR e OPT", flush=True)
 
    # split 80/20
    random.seed(42)
    shuffled_ids = train_data_IDS.copy()
    random.shuffle(shuffled_ids)

    split_idx = int(len(shuffled_ids) * 0.8)
    train_ids = shuffled_ids[:split_idx]
    test_ids = shuffled_ids[split_idx:]

    print(f"serie train 80%: {len(train_ids)}", flush=True)
    print(f"serie train 20%: {len(test_ids)} ", "\n", flush=True)
 
 
    # salvataggio log liste train e test
    with open(f'train_ids_{job_name}.json', 'w') as f:
        json.dump(train_ids, f)
    try:
        project_work.log_artifact(name=f"moco2D_train-ids_{job_name}", source=f'train_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload train_ids: {e}", flush=True)

    with open(f'test_ids_{job_name}.json', 'w') as f:
        json.dump(test_ids, f)
    try:
        project_work.log_artifact(name=f"moco2D_test-ids_{job_name}", source=f'test_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload test: {e}", flush=True)

    # caricamento pesi encoders
    try:
        path_sar = project_work.get_artifact(f"autoencoder2D_SAR_weights_{weights_encoder_sar}").download(f"autoencoder2D_SAR_model_best_{weights_encoder_sar}.pth")
        path_opt = project_work.get_artifact(f"autoencoder2D_OPT_weights_{weights_encoder_opt}").download(f"autoencoder2D_OPT_model_best_{weights_encoder_opt}.pth")

        modelSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=attn).to(device)
        modelSAR.load_state_dict(torch.load(path_sar, map_location=device))

        modelOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=attn).to(device)
        modelOPT.load_state_dict(torch.load(path_opt, map_location=device))
        print(f"OK -> Pesi encoders caricati: {weights_encoder_sar}, {weights_encoder_opt}", flush=True)
    except Exception as e:
        print(f"EXC -> upload pesi encoders: {e}", flush=True)  


    # TRAINING MOCO 2D
    
    print("--- Training MoCo 2D ---")
    model = MoCo2encoders_2d(
        base_encoder_q=modelOPT.encoder,
        base_encoder_k=modelSAR.encoder,
        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
        symmetric=symmetric, device=device,
        hidden_channels_dim=hidden_channels_dim,
        simple_proj=simple_proj,
        attn=attn
    ).to(device)
 
    results = {'lr': [], 'train_loss': []}
    best_loss = float('inf')
    epochs_no_improve = 0

    # variabili per caricamento pesi
    start_epoch = 1
    queue_restored = False

    # resume = True -> carico pesi vecchi di moco e coda già inzializzata
    if resume:
        try:
            w_path = project_work.get_artifact(f"moco2D_weights_{job_name}").download("/data")
            state_dict = torch.load(w_path, map_location=device)
            model.load_state_dict(state_dict)
            queue_restored = True
            print("OK -> pesi MoCo caricati", flush=True)
        except Exception as e:
            print(f"EXC -> no pesi MoCo vecchi, inizializzazione casuale: {e}", flush=True)
 
        try:
            m_path = project_work.get_artifact(f"moco2D_metrics_{job_name}").download("/data")
            prev_df = pd.read_csv(m_path)
            best_loss = prev_df['train_loss'].min()
            results = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}

            # se carico pesi conteggio riparte da ultima epoca
            start_epoch = len(prev_df) + 1
            print(f"OK -> metriche MoCo caricate, best_loss={best_loss}", flush=True)
        except Exception as e:
            print(f"EXC -> nessuna metrica MoCo trovata: {e}", flush=True)
 
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
 
    # prima epoca loss molto bassa perche coda inizializzata casualmente
    # lr parte da 0 e poi si stabilizza al valore fissato
    # con batch size = 16 e coda (moco_k) = 4096, la coda si riempie con 256 batch
    # una epoca contiene 1243 batch

    epoch_lenght = None
    base_lr = lr
    batch_step = 0
 
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
        print(f"EXC -> MoCo2encodersLoader: {e}", flush=True)
 
    try:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True,
        )
        # quanti batch stanno in un epoca, per Standard e barch size 16 = 1243
        epoch_lenght = len(train_loader) if not queue_restored else None 
    except Exception as e:
        print(f"EXC -> DataLoader: {e}", flush=True)

    # libreria time utilizzata per debug, tempo training
    try:
        for epoch in range(start_epoch, epochs + 1):  
            model.train()
            model.encoder_q.eval()
            model.encoder_k.eval()
 
            total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
 
            if time_debug:
                t_prev = time.time()
 
            for im_q, im_k in train_bar:

                # lr sale da 0 a lr incrementalmente ad ogni batch della prima epoca
                # quindi da 0 a 1243, da 1244 lr diventa il paramentro passato nel notebook

                if epoch_lenght is not None and batch_step < epoch_lenght:
                    batch_lr = base_lr * (batch_step + 1) / epoch_lenght
                    for pg in optimizer.param_groups:
                        pg['lr'] = batch_lr
                batch_step += 1
 
                if time_debug:
                    torch.cuda.synchronize()
                    t_data = time.time()
 
                im_q = im_q.to(non_blocking=True, device=device)
                im_k = im_k.to(non_blocking=True, device=device)
 
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
                    print(f"data: {t_data-t_prev:.3f}s | transfer: {t_transfer-t_data:.3f}s | "
                        f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                    t_prev = time.time()
  
                total_num += batch_size
                total_loss += loss.item() * batch_size
                train_bar.set_description(f'MoCo Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')
 
            epoch_loss = total_loss / total_num
            results['lr'].append(optimizer.param_groups[0]['lr'])
            results['train_loss'].append(epoch_loss)
 
            # loss migliorata -> salva metriche e pesi
            pd.DataFrame(results).to_csv(f'moco2D_log_{job_name}.csv', index_label='epoch')

            # refresh token e progetto
            try:
                dh.refresh_token()
                print("OK -> refresh token", flush=True)
            except Exception as e:
                print(f"EXC -> refresh token: {e}", flush=True)

            try:
                project_work = dh.get_project("floods")
                print("OK -> get project\n", flush=True)
            except Exception as e:
                print(f"EXC -> get project: {e}", "\n", flush=True)            

            try:
                project_work.log_artifact(name=f"moco2D_metrics_{job_name}", source=f'moco2D_log_{job_name}.csv', kind='artifact')
                print(f"OK -> metriche MoCo salvate", flush=True)
            except Exception as e:
                print(f"EXC -> upload metriche MoCo: {e}", flush=True)
 
            # skip first epoch for patience
            skip_patience_epoch = (epoch == start_epoch) and (not queue_restored)
 
            if skip_patience_epoch:
                torch.save(model.state_dict(), f'moco2D_model_best_{job_name}.pth')
                print(f"skip patience epoch {epoch}", flush=True)
                print(f"loss skipped epoch {epoch}: {best_loss}", flush=True)
 
                try:
                    project_work.log_artifact(name=f"moco2D_weights_{job_name}", source=f'moco2D_model_best_{job_name}.pth', kind='artifact')
                    print(f"OK -> pesi salvati\n", flush=True)
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", "\n", flush=True)
 
            elif epoch_loss < best_loss - min_delta:
                torch.save(model.state_dict(), f'moco2D_model_best_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0
                print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)
 
                try:
                    project_work.log_artifact(name=f"moco2D_weights_{job_name}", source=f'moco2D_model_best_{job_name}.pth', kind='artifact')
                    print(f"OK -> pesi salvati", flush=True)
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", flush=True)
 
            else:
                epochs_no_improve += 1
                print(f"loss epoch {epoch} : {epoch_loss}", flush=True)
                print(f"best loss epoch {epoch} (old): {best_loss}", flush=True)
 
                if epochs_no_improve >= patience:
                    print(f"Early Stopping MoCo: epoch {epoch} -> best loss: {best_loss}", "\n", flush=True)
                    break
 
    except Exception as e:
        print(f"EXC -> training MoCo: {e}", "\n", flush=True)
 
    print(f"OK -> Terminato training MoCo, LOSS (InfoNCE): {best_loss}", "\n", flush=True)
    
    return "TERMINATO -> training moco2D"



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
    weights_sar: str = "weights_s1",
    weights_opt: str = "weights_s2",
    n_samples: int = 10,
    job_name: str = "nome",
    ltae: bool = False,
    monodimensional: bool = False,
    recon_channels: list | None = None,
    save_dir: str = "/data/debug_recon",
):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
 
    torch.backends.cudnn.benchmark = True
 
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
    
    sar_zip_map = {}
    opt_zip_map = {}
    train_data_SAR_IDS = []
    train_data_OPT_IDS = []
    
 
    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")
    print("OK -> Download terminato", flush=True)

    save_dir = save_dir + "_" + job_name
    os.makedirs(save_dir, exist_ok=True)


    try:
        if test_sar:
            sar_dir = os.path.join(dataset_path, "SAR")
            sar_zip_map = {}
            for zip_file in sorted(glob(os.path.join(sar_dir, "SAR_*.zip"))):
                with zipfile.ZipFile(zip_file, 'r') as z:
                    for n in z.namelist():
                        if n.lower().endswith('.tif'):
                            ID = os.path.basename(n).split('_SAR_')[0]
                            sar_zip_map[ID] = zip_file
            train_data_SAR_IDS = list(sar_zip_map.keys())
            random_sar_ids = random.sample(train_data_SAR_IDS, min(n_samples, len(train_data_SAR_IDS))) if train_data_SAR_IDS else []
            print(f"{len(train_data_SAR_IDS)} serie SAR trovate", flush=True)
 
        if test_opt:
            opt_dir = os.path.join(dataset_path, "OPT")
            opt_zip_map = {}
            for zip_file in sorted(glob(os.path.join(opt_dir, "OPT_*.zip"))):
                with zipfile.ZipFile(zip_file, 'r') as z:
                    for n in z.namelist():
                        if n.lower().endswith('.tif'):
                            ID = os.path.basename(n).split('_OPT_')[0]
                            opt_zip_map[ID] = zip_file
            train_data_OPT_IDS = list(opt_zip_map.keys())
            random_opt_ids = random.sample(train_data_OPT_IDS, min(n_samples, len(train_data_OPT_IDS))) if train_data_OPT_IDS else []
            print(f"{len(train_data_OPT_IDS)} serie OPT trovate", flush=True)
 
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
 
            datasetSAR = Singlemodal_Loader(
                listIDs=random_sar_ids, root=dataset_path, zip_map=sar_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images1,
                n_channels=n_channels1, data_type='SAR'
            )
            loaderSAR = torch.utils.data.DataLoader(datasetSAR, batch_size=1, shuffle=False)
 
            print("SERIE SAR:", flush=True)
            with torch.no_grad():
                for i, im in enumerate(loaderSAR):
                    im = im.to(device)

                    if monodimensional:
                        try:
                            latent_vector = modelSAR.encoder(im)
                        except AttributeError as e:
                            print(f"EXC -> nome encoder sar: {e}", flush=True)
                            break    
 
                        vector = latent_vector.cpu().numpy().flatten()
                        print(f"ID {i}: {random_sar_ids[i]} | shape: {list(latent_vector.shape)}", flush=True)
                        print(f"VEC: {np.round(vector, 4)}\n", flush=True)
 
                    save_reconstruction_pngs(
                        modelSAR, im, save_dir=save_dir,
                        prefix=f"SAR_{random_sar_ids[i]}",
                        channels=recon_channels, device=device,
                    )
 
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
 
            datasetOPT = Singlemodal_Loader(
                listIDs=random_opt_ids, root=dataset_path, zip_map=opt_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images2,
                n_channels=n_channels2, data_type='OPT'
            )
            loaderOPT = torch.utils.data.DataLoader(datasetOPT, batch_size=1, shuffle=False)
 
            print("SERIE OPT:", flush=True)
            with torch.no_grad():
                for i, im in enumerate(loaderOPT):
                    im = im.to(device)

                    if monodimensional:
                        try:
                            latent_vector = modelOPT.encoder(im)
                        except AttributeError as e:
                            print(f"EXC -> nome encoder opt: {e}", flush=True)
                            break
 
                        vector = latent_vector.cpu().numpy().flatten()
                        print(f"ID {i}: {random_opt_ids[i]} | shape: {list(latent_vector.shape)}", flush=True)
                        print(f"VEC: {np.round(vector, 4)}\n", flush=True)
 
                    save_reconstruction_pngs(
                        modelOPT, im, save_dir=save_dir,
                        prefix=f"OPT_{random_opt_ids[i]}",
                        channels=recon_channels, device=device,
                    )
 
            del modelOPT
            torch.cuda.empty_cache()
 
    # zip png, upload artifact
    zip_base = save_dir.rstrip("/")
    zip_path = f"{zip_base}.zip"
    try:
        shutil.make_archive(zip_base, 'zip', save_dir)
        project_work.log_artifact(
            name=f"test-encoders-reconstructions_{dataset}_{job_name}",
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


