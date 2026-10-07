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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import json
import time
import zipfile
import torchvision.transforms as transforms
import math
import re
import xarray as xr
import zarr
import traceback
import tarfile
import io
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from digitalhub_runtime_python import handler

from csv.src.multimodal_3Dconv_attention import Singlemodal_CAE
from csv.src.multimodal_3Dconv_attention import Singlemodal_CAE_2d
from csv.src.moco.loader import Singlemodal_Loader
from csv.src.moco.loader import MoCo2encodersLoader
from csv.src.moco.builder import MoCo2encoders
from csv.src.moco.builder import MoCo2encoders_2d

matplotlib.use("Agg")  


# notebook -> autoencoder1D

@handler()
def train_autoencoder_1D(
    job_name: str = "check_sar_v1",
    dataset: str = "Test",
    sensor: str = "SAR",
    resume: bool = True,
    time_debug: bool = False,

    epochs: int = 1, 
    batch_size: int = 1, 
    lr: float = 1e-4, 
    weight_decay: float = 1e-4,
    patch_size: int = 256,
    n_images: int = 4,
    n_channels: int = 2,
    latent_space_vector_dim: int = 512,
    workers: int = 0,
    patience: int = 20,
    min_delta: float = 1e-4    
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, flush=True)
 
    n_gpus = torch.cuda.device_count()
    print(f"GPU: {n_gpus}", "\n", flush=True)
 
    torch.backends.cudnn.benchmark = True
 
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    zip_map = {}
    zip_map = {}
    train_data_IDS = []
   
    # download dataset
    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}_crop_norm").download("/data/dataset_floods")
    print("OK -> Download terminato\n")

    try:
        os.makedirs(f"/data/cache_{sensor}", exist_ok=True)
        part_files = sorted(glob(os.path.join(dataset_path, sensor, "part*.zip")))
        if part_files:
            for part_path in part_files:
                with zipfile.ZipFile(part_path, 'r') as z:
                    z.extractall(f"/data/cache_{sensor}")
            train_data_IDS = [f[:-4] for f in os.listdir(f"/data/cache_{sensor}") if f.endswith('.npy')]
            print(f"OK -> caricate {len(train_data_IDS)} serie {sensor}", flush=True)
    except Exception as e:
        print(f"EXC -> caricamento serie {sensor}: {e}", flush=True)     


    # TRAINING AUTOENCODER

    print(f"--- Training {sensor} ---", "\n", flush=True)
    model = Singlemodal_CAE(input_dim=n_channels, output_dim=latent_space_vector_dim, n_images=n_images).to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss().to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')

    results = {'lr': [], 'train_loss': []}

    # resume = True, carica pesi e metriche vecchio train
    if resume:
        try:
            w_path = project_work.get_artifact(f"autoencoder_1D_{sensor}_weights_{job_name}").download("/data")
            state_dict = torch.load(w_path, map_location=device)
            (model.module if n_gpus > 1 else model).load_state_dict(state_dict)
            print(f"OK -> pesi {sensor} caricati: {job_name}", flush=True)
        except Exception as e:
            print(f"EXC -> nessun peso {sensor} trovato: inizializzazione casuale: {e}", flush=True)

        try:
            m_path = project_work.get_artifact(f"autoencoder_1D_{sensor}_metrics_{job_name}").download("/data")
            prev_df = pd.read_csv(m_path)
            best_loss = prev_df['train_loss'].min()
            results = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
            print(f"OK -> metriche {sensor} caricate: best_loss={best_loss}", flush=True)
        except Exception as e:
            print(f"EXC -> nessuna metrica {sensor} trovata: {e}", flush=True)

        print("\n")    

    try:
        dataset = Singlemodal_Loader(
            listIDs=train_data_IDS, root=dataset_path, zip_map=zip_map,
            transform=None, patch_size=patch_size, n_images=n_images,
            n_channels=n_channels, data_type=sensor
        )
        for zip_path in set(zip_map.values()):
            if os.path.exists(zip_path):
                os.remove(zip_path)
        print(f"OK -> Zip {sensor} eliminati\n", flush=True)
    except Exception as e:
        print(f"EXC -> Loader {sensor}:", {e}, "\n", flush=True)

    # check per mse loss
    # calcolo della mse se l'autoencoder producesse sempre la media -> mean_mse_loss
    # train loss buona se inferiose alla mean_mse_loss
    try:
        sample_ids = train_data_IDS[:100]
        mean_values = None
        n = 0

        for ID in sample_ids:
            im = np.load(os.path.join(dataset.cache_dir, f"{ID}.npy"))
            if mean_values is None:
                mean_values = np.zeros_like(im, dtype=np.float64)
            mean_values += im
            n += 1
        mean_image = (mean_values / n).astype(np.float32)

        total_squared_error = 0.0
        total_values = 0
        for ID in sample_ids:
            im = np.load(os.path.join(dataset.cache_dir, f"{ID}.npy"))
            total_squared_error += np.sum((im - mean_image) ** 2)
            total_values += im.size

        mean_mse_loss = total_squared_error/total_values     

        print(f"mean mse loss {sensor}: {mean_mse_loss:.6f}", "\n", flush=True)
    except Exception as e:
        print(f"EXC -> mean mse loss {sensor}: {e}", "\n", flush=True)    

    try:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    except Exception as e:
        print(f"EXC -> DataLoader {sensor} {e}", flush=True)

    epochs_no_improve = 0

    # libreria time utilizzata per debug, tempo training
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloader, mininterval=5.0)

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
                    output = model(im)
                    loss = loss_fn(output, im)

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
                    print(f"dati: {t_data-t_prev:.3f}s | transfer: {t_transfer-t_data:.3f}s | "
                        f"forward: {t_forward-t_transfer:.3f}s | backward: {t_backward-t_forward:.3f}s", flush=True)
                    t_prev = time.time()

                total_num += dataloader.batch_size
                total_loss += loss.item() * dataloader.batch_size
                train_bar.set_description(f'{sensor} Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')

            epoch_loss = total_loss / total_num
            results['lr'].append(optimizer.param_groups[0]['lr'])
            results['train_loss'].append(epoch_loss)

            pd.DataFrame(results).to_csv(f'autoencoder_1D_{sensor}_log_{job_name}.csv', index_label='epoch')

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

            # salvataggio metriche
            try:    
                project_work.log_artifact(name=f"autoencoder_1D_{sensor}_metrics_{job_name}", source=f'autoencoder_1D_{sensor}_log_{job_name}.csv', kind='artifact')
                print(f"OK -> metriche {sensor} salvate", flush=True)
            except Exception as e:
                print(f"EXC -> salvataggio metriche {sensor}: {e}", flush=True)                

            if epoch_loss < best_loss - min_delta:
                state_dict = model.module.state_dict() if n_gpus > 1 else model.state_dict()
                torch.save(state_dict, f'autoencoder_1D_{sensor}_model_best_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0
                print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                # salvataggio pesi
                try:
                    project_work.log_artifact(name=f"autoencoder_1D_{sensor}_weights_{job_name}", source=f'autoencoder_1D_{sensor}_model_best_{job_name}.pth', kind='artifact')
                    print(f"OK -> pesi {sensor} salvati\n", flush=True)
                except Exception as e:
                    print(f"EXC -> salvataggio pesi {sensor}: {e}", "\n", flush=True)

            else:
                epochs_no_improve += 1
                print(f"loss epoch {epoch}: {epoch_loss}", flush=True)
                print(f"best loss epoch {epoch} (old): {best_loss}", "\n", flush=True)

                if epochs_no_improve >= patience:
                    print(f"Early Stopping epoch {epoch} -> best loss: {best_loss}", "\n", flush=True)
                    break

    except Exception as e:
        print(f"EXC -> training {sensor}: {e}", "\n", flush=True)

    print(f"OK -> Terminato training {sensor}, LOSS (MSE): {best_loss}", "\n", flush=True)     
  
    return "TERMINATO -> training autoencoder 1D"


# notebook -> autoencoder2D

@handler()
def train_autoencoder_2D(
    job_name: str = "nome_job",
    dataset: str = "Test",
    sensor: str = "SAR",
    ltae: bool = False,
    resume: bool = True,
    time_debug: bool = False,

    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patch_size: int = 256,
    n_images: int = 4,
    n_channels: int = 2,
    num_channels_latent_space: int = 16,
    workers: int = 0,
    patience: int = 20,
    min_delta: float = 1e-4,
    
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, flush=True)
 
    n_gpus = torch.cuda.device_count()
    print(f"GPU: {n_gpus}", "\n", flush=True)
 
    torch.backends.cudnn.benchmark = True
 
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
    zip_map = {}
    zip_map = {}
    train_data_IDS = []
   
    # download dataset
    print(f"Download: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}_crop_norm").download("/data/dataset_floods")
    print("OK -> Download terminato\n")

    try:
        os.makedirs(f"/data/cache_{sensor}", exist_ok=True)
        part_files = sorted(glob(os.path.join(dataset_path, sensor, "part*.zip")))
        if part_files:
            for part_path in part_files:
                with zipfile.ZipFile(part_path, 'r') as z:
                    z.extractall(f"/data/cache_{sensor}")
            train_data_IDS = [f[:-4] for f in os.listdir(f"/data/cache_{sensor}") if f.endswith('.npy')]
            print(f"OK -> caricate {len(train_data_IDS)} serie {sensor}", flush=True)
    except Exception as e:
        print(f"EXC -> caricamento serie {sensor}: {e}", flush=True)    
 
    
    # TRAINING AUTOENCODER
    
    print(f"--- Training {sensor} ---", "\n", flush=True)
    model = Singlemodal_CAE_2d(input_dim=n_channels, output_dim=num_channels_latent_space, n_images=n_images, n_head=8, d_k=8, ltae=ltae).to(device)
    
    if n_gpus > 1:
        model= nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss().to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')

    results = {'lr': [], 'train_loss': []}

    # resume = True, carica pesi e metriche vecchio train
    if resume:
        try:
            w_path = project_work.get_artifact(f"autoencoder_2D_{sensor}_weights_{job_name}").download("/data")
            state_dict = torch.load(w_path, map_location=device)
            (model.module if n_gpus > 1 else model).load_state_dict(state_dict)
            print(f"OK -> pesi {sensor} caricati -> {job_name}", flush=True)
        except Exception as e:
            print(f"EXC -> nessun peso {sensor} trovato: inizializzazione casuale: {e}", flush=True)

        try:
            m_path = project_work.get_artifact(f"autoencoder_2D_{sensor}_metrics_{job_name}").download("/data")
            prev_df = pd.read_csv(m_path)
            best_loss = prev_df['train_loss'].min()
            results = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
            print(f"OK -> metriche {sensor} caricate: best_loss={best_loss}", flush=True)
        except Exception as e:
            print(f"EXC -> nessuna metrica {sensor} trovata: {e}", flush=True)

        print("\n")    

    try:
        dataset = Singlemodal_Loader(
            listIDs=train_data_IDS, root=dataset_path, zip_map=zip_map,
            transform=None, patch_size=patch_size, n_images=n_images,
            n_channels=n_channels, data_type=sensor
        )
        for zip_path in set(zip_map.values()):
            if os.path.exists(zip_path):
                os.remove(zip_path)
        print(f"OK -> Zip {sensor} eliminati\n", flush=True)
    except Exception as e:
        print(f"EXC -> Loader {sensor}:", {e}, "\n", flush=True)

    # check per mse loss
    # calcolo della mse se l'autoencoder producesse sempre la media -> mean_mse_loss
    # train loss buona se inferiose alla mean_mse_loss
    try:
        sample_ids = train_data_IDS[:100]
        mean_values = None
        n = 0

        for ID in sample_ids:
            im = np.load(os.path.join(dataset.cache_dir, f"{ID}.npy"))
            if mean_values is None:
                mean_values = np.zeros_like(im, dtype=np.float64)
            mean_values += im
            n += 1
        mean_image = (mean_values / n).astype(np.float32)

        total_squared_error = 0.0
        total_values = 0
        for ID in sample_ids:
            im = np.load(os.path.join(dataset.cache_dir, f"{ID}.npy"))
            total_squared_error += np.sum((im - mean_image) ** 2)
            total_values += im.size

        mean_mse_loss = total_squared_error/total_values     

        print(f"mean mse loss {sensor}: {mean_mse_loss:.6f}", "\n", flush=True)
    except Exception as e:
        print(f"EXC -> mean mse loss {sensor}: {e}", "\n", flush=True)


    try:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    except Exception as e:
        print(f"EXC -> DataLoader {sensor} {e}", flush=True)

    epochs_no_improve = 0

    # libreria time utilizzata per debug, tempo training
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloader, mininterval=5.0)

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
                    output = model(im)
                    loss = loss_fn(output, im)

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

                total_num += dataloader.batch_size
                total_loss += loss.item() * dataloader.batch_size
                train_bar.set_description(f'{sensor} Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')

            epoch_loss = total_loss / total_num
            results['lr'].append(optimizer.param_groups[0]['lr'])
            results['train_loss'].append(epoch_loss)

            # loss migliorata -> salva metriche e pesi
            pd.DataFrame(results).to_csv(f'autoencoder_2D_{sensor}_log_{job_name}.csv', index_label='epoch')

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
                project_work.log_artifact(name=f"autoencoder_2D_{sensor}_metrics_{job_name}", source=f'autoencoder_2D_{sensor}_log_{job_name}.csv', kind='artifact')
                print(f"OK -> metriche {sensor} salvate", flush=True)
            except Exception as e:
                print(f"EXC -> salvataggio metriche {sensor}: {e}", flush=True)                

            if epoch_loss < best_loss - min_delta:
                state_dict = model.module.state_dict() if n_gpus > 1 else model.state_dict()
                torch.save(state_dict, f'autoencoder_2D_{sensor}_model_best_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0
                print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                try:
                    project_work.log_artifact(name=f"autoencoder_2D_{sensor}_weights_{job_name}", source=f'autoencoder_2D_{sensor}_model_best_{job_name}.pth', kind='artifact')
                    print(f"OK -> pesi {sensor} salvati\n", flush=True)
                except Exception as e:
                    print(f"EXC -> salvataggio pesi {sensor}: {e}", "\n", flush=True)

            else:
                epochs_no_improve += 1
                print(f"loss epoch {epoch}: {epoch_loss}", flush=True)
                print(f"best loss epoch {epoch} (old): {best_loss}", "\n", flush=True)

                if epochs_no_improve >= patience:
                    print(f"Early Stopping epoch {epoch} -> best loss: {best_loss}", "\n", flush=True)
                    break


    except Exception as e:
        print(f"EXC -> training {sensor}: {e}", "\n", flush=True)

    print(f"OK -> Terminato training {sensor}, LOSS (MSE): {best_loss}",  "\n", flush=True) 
 
    return "TERMINATO -> Training autoencoder 2D"


# notebook -> moco1D

handler()
def train_moco_1D(
    job_name: str = "nome_job",
    dataset: str = "Test",
    weights_encoder_sar: str = "train_sar_1D_v1_Standard_200",
    weights_encoder_opt: str = "train_opt_1D_v1_Standard_200",
    resume: bool = True,
    time_debug: bool = False,

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
    patience: int = 20,
    min_delta: float = 1e-4,

):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

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
        project_work.log_artifact(name=f"moco_1D_trainIdsList_{job_name}", source=f'train_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload train_ids: {e}", flush=True)

    with open(f'test_ids_{job_name}.json', 'w') as f:
        json.dump(test_ids, f)
    try:
        project_work.log_artifact(name=f"moco_1D_testIdsList_{job_name}", source=f'test_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload test: {e}", flush=True)

    # caricamento pesi encoders
    try:
        path_sar = project_work.get_artifact(weights_encoder_sar).download(f"{weights_encoder_sar}.pth")
        path_opt = project_work.get_artifact(weights_encoder_opt).download(f"{weights_encoder_opt}.pth")

        modelSAR = Singlemodal_CAE(input_dim=n_channels1, output_dim=512, n_images=n_images1).to(device)
        modelSAR.load_state_dict(torch.load(path_sar, map_location=device))

        modelOPT = Singlemodal_CAE(input_dim=n_channels2, output_dim=512, n_images=n_images2).to(device)
        modelOPT.load_state_dict(torch.load(path_opt, map_location=device))
        print(f"OK -> Pesi encoders caricati: {weights_encoder_sar}, {weights_encoder_opt}",  "\n", flush=True)
    except Exception as e:
        print(f"EXC -> upload pesi encoders: {e}",  "\n", flush=True)   


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
            w_path = project_work.get_artifact(f"moco_1D_weights_{job_name}").download("/data")
            state_dict = torch.load(w_path, map_location=device)
            model.load_state_dict(state_dict)
            queue_restored = True
            print("OK -> pesi MoCo caricati", flush=True)
        except Exception as e:
            print(f"EXC -> no pesi MoCo vecchi, inizializzazione casuale: {e}", flush=True)
 
        try:
            m_path = project_work.get_artifact(f"moco_1D_metrics_{job_name}").download("/data")
            prev_df = pd.read_csv(m_path)
            best_loss = prev_df['train_loss'].min()
            results = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}

            # se carico pesi conteggio riparte da ultima epoca, best loss esclude riga 0 del csv
            start_epoch = len(prev_df) + 1
            valid_losses = prev_df['train_loss'].iloc[1:]
            best_loss = valid_losses.min() if len(valid_losses) > 0 else float('inf')

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
            pd.DataFrame(results).to_csv(f'moco_1D_log_{job_name}.csv', index_label='epoch')

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
                project_work.log_artifact(name=f"moco_1D_metrics_{job_name}", source=f'moco_1D_log_{job_name}.csv', kind='artifact')
                print(f"OK -> metriche MoCo salvate", flush=True)
            except Exception as e:
                print(f"EXC -> upload metriche MoCo: {e}", flush=True)

            # skip first epoch for patience
            skip_patience_epoch = (epoch == start_epoch) and (not queue_restored)
 
            if skip_patience_epoch:
                torch.save(model.state_dict(), f'moco_1D_model_best_{job_name}.pth')
                print(f"skip patience epoch {epoch}", "\n", flush=True)   

                try:
                    project_work.log_artifact(name=f"moco_1D_weights_{job_name}", source=f'moco_1D_model_best_{job_name}', kind='artifact')
                    print(f"OK -> pesi salvati", flush=True)
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", flush=True)                             

            elif epoch_loss < best_loss - min_delta:
                torch.save(model.state_dict(), f'moco_1D_model_best_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0
                print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                try:
                    project_work.log_artifact(name=f"moco_1D_weights_{job_name}", source=f'moco_1D_model_best_{job_name}', kind='artifact')
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
    
    return "TERMINATO -> training moco 1D"


# notebook -> moco2D

@handler()
def train_moco_2D(
    job_name: str = "nome_job",
    dataset: str = "Test",
    weights_encoder_sar: str = "train_sar_1D_v1_Standard_200",
    weights_encoder_opt: str = "train_opt_1D_v1_Standard_200",
    hidden_channels_dim: int = 64,
    simple_proj: bool = True,
    attn: bool = False,
    pool_grid: int = 1,
    resume: bool = True,  
    lr_min: float = 0.0003,   
    cosine: bool = False,    
    time_debug: bool = False,
 
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
    patience: int = 20,
    min_delta: float = 1e-4,
    
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
        project_work.log_artifact(name=f"moco_2D_trainIdsList_{job_name}", source=f'train_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload train_ids: {e}", flush=True)
 
    with open(f'test_ids_{job_name}.json', 'w') as f:
        json.dump(test_ids, f)
    try:
        project_work.log_artifact(name=f"moco_2D_testIdsList_{job_name}", source=f'test_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload test: {e}", flush=True)
 
    # caricamento pesi encoders
    try:
        path_sar = project_work.get_artifact(weights_encoder_sar).download(f"{weights_encoder_sar}.pth")
        path_opt = project_work.get_artifact(weights_encoder_opt).download(f"{weights_encoder_opt}.pth")
 
        modelSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=attn).to(device)
        modelSAR.load_state_dict(torch.load(path_sar, map_location=device))
 
        modelOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=attn).to(device)
        modelOPT.load_state_dict(torch.load(path_opt, map_location=device))
        print(f"OK -> Pesi encoders caricati: {weights_encoder_sar}, {weights_encoder_opt}", "\n", flush=True)
    except Exception as e:
        print(f"EXC -> upload pesi encoders: {e}", "\n", flush=True)  
 
 
    # TRAINING MOCO 2D
    
    print("--- Training MoCo 2D ---")
    model = MoCo2encoders_2d(
        base_encoder_q=modelOPT.encoder,
        base_encoder_k=modelSAR.encoder,
        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
        symmetric=symmetric, device=device,
        hidden_channels_dim=hidden_channels_dim,
        pool_grid=pool_grid,
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
            w_path = project_work.get_artifact(f"moco_2D_weights_{job_name}").download("/data")
            state_dict = torch.load(w_path, map_location=device)
            model.load_state_dict(state_dict)
            queue_restored = True
            print("OK -> pesi MoCo caricati", flush=True)
        except Exception as e:
            print(f"EXC -> no pesi MoCo vecchi, inizializzazione casuale: {e}", flush=True)
 
        try:
            m_path = project_work.get_artifact(f"moco_2D_metrics_{job_name}").download("/data")
            prev_df = pd.read_csv(m_path)
            best_loss = prev_df['train_loss'].min()
            results = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
 
            # se carico pesi conteggio riparte da ultima epoca, best loss esclude riga 0 del csv
            start_epoch = len(prev_df) + 1
            valid_losses = prev_df['train_loss'].iloc[1:]
            best_loss = valid_losses.min() if len(valid_losses) > 0 else float('inf')
 
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
    steps_total = None   # NUOVO: step totali del run, serve al coseno
    base_lr = lr
    batch_step = 0

    target_epoch = start_epoch + epochs - 1
 
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
        # quanti run farebbero se tutte le epoche vengono eseguite, no early stopping
        epoch_lenght = len(train_loader) if not queue_restored else None 
        steps_total = (target_epoch - start_epoch + 1) * len(train_loader)

    except Exception as e:
        print(f"EXC -> DataLoader: {e}", flush=True)    
 
    # libreria time utilizzata per debug, tempo training
    try:
        for epoch in range(start_epoch, target_epoch + 1):  
            model.train()
            model.encoder_q.eval()
            model.encoder_k.eval()
 
            total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
 
            if time_debug:
                t_prev = time.time()
 
            for im_q, im_k in train_bar:
 
                # lr sale da 0 a lr incrementalmente ad ogni batch della prima epoca
                # quindi da 0 a 1243, da 1244 lr diventa il paramentro passato nel notebook
                # cosine = True, dopo 1244 lr scende a coseno da lr a lr_min
                # cosine = False, lr rimane parametro passato al notebook
 
                if epoch_lenght is not None and batch_step < epoch_lenght:
                    batch_lr = base_lr * (batch_step + 1) / epoch_lenght
                    for pg in optimizer.param_groups:
                        pg['lr'] = batch_lr
                elif cosine and steps_total is not None:
                    w = epoch_lenght or 0
                    progress = (batch_step - w) / max(1, steps_total - w)
                    batch_lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * progress))
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
                train_bar.set_description(f'MoCo Epoch: [{epoch}/{target_epoch}], Loss: {loss.item():.4f}')
 
            epoch_loss = total_loss / total_num
            results['lr'].append(optimizer.param_groups[0]['lr'])
            results['train_loss'].append(epoch_loss)
 
            # loss migliorata -> salva metriche e pesi
            pd.DataFrame(results).to_csv(f'moco_2D_log_{job_name}.csv', index_label='epoch')
 
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
                project_work.log_artifact(name=f"moco_2D_metrics_{job_name}", source=f'moco_2D_log_{job_name}.csv', kind='artifact')
                print(f"OK -> metriche MoCo salvate", flush=True)
            except Exception as e:
                print(f"EXC -> upload metriche MoCo: {e}", flush=True)
 
            # skip first epoch for patience
            skip_patience_epoch = (epoch == start_epoch) and (not queue_restored)
 
            if skip_patience_epoch:
                torch.save(model.state_dict(), f'moco_2D_model_best_{job_name}.pth')
                print(f"skip patience epoch {epoch}", flush=True)
                print(f"loss skipped epoch {epoch}: {best_loss}", flush=True)
 
                try:
                    project_work.log_artifact(name=f"moco_2D_weights_{job_name}", source=f'moco_2D_model_best_{job_name}.pth', kind='artifact')
                    print(f"OK -> pesi salvati\n", flush=True)
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", "\n", flush=True)
 
            elif epoch_loss < best_loss - min_delta:
                torch.save(model.state_dict(), f'moco_2D_model_best_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0
                print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)
 
                try:
                    project_work.log_artifact(name=f"moco_2D_weights_{job_name}", source=f'moco_2D_model_best_{job_name}.pth', kind='artifact')
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
    
    return "TERMINATO -> training moco 2D"



@handler()
def rename_artifact(
    old_metrics: str = "old_metrics",
    old_weights: str = "old_weights",
    new_metrics: str = "new_metrics",
    new_weights: str = "new_weights"
):
    project_work = dh.get_project("floods")

    # pesi
    local_path = project_work.get_artifact(old_weights).download("/data/tmp_weights.pth")
    project_work.log_artifact(name=new_weights, source=local_path, kind="artifact")

    # metriche
    local_path_metrics = project_work.get_artifact(old_metrics).download("/data/tmp_metrics.csv")
    project_work.log_artifact(name=new_metrics, source=local_path_metrics, kind="artifact")
    return


@handler()
def eval_moco_retrieval(
    job_name: str = "nome_job",         
    dataset: str = "Standard",
    simple_proj: bool = False,           
    attn: bool = False,                  
    hidden_channels_dim: int = 64,
    moco_dim: int = 128,
    moco_k: int = 4096,
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    patch_size: int = 256,
    pool_grid: int = 1,
    n_images1: int = 4, n_channels1: int = 2,
    n_images2: int = 4, n_channels2: int = 10,     
    tile_size_m: float = 2560.0,         
    batch_size: int = 32,
    workers: int = 0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
 
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
 
 
    try:
        ids_path = project_work.get_artifact(f"moco_2D_testIdsList_{job_name}").download("/data/eval_test_ids.json", overwrite=True)
        with open(ids_path, 'r') as f:
            eval_ids = json.load(f)
        print(f"OK -> {len(eval_ids)} serie testIdsList caricate", flush=True)
    except Exception as e:
        print(f"EXC -> caricamento lista testIdsList: {e}", flush=True) 
 
    print(f"Download dataset: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}_crop_norm").download("/data/dataset_floods")
    print("OK -> Download terminato", flush=True)
 
    needed = {f"{ID}.npy" for ID in eval_ids}
    for modality, cache_dir in [("SAR", "/data/cache_SAR"), ("OPT", "/data/cache_OPT")]:
        os.makedirs(cache_dir, exist_ok=True)
        part_files = sorted(glob(os.path.join(dataset_path, modality, "part*.zip")))
        for part_path in part_files:
            with zipfile.ZipFile(part_path, 'r') as z:
                for name in z.namelist():
                    if name in needed:
                        z.extract(name, cache_dir)
            os.remove(part_path) 
        print(f"OK -> {modality}: serie testIdsList estratte", flush=True)
 
    eval_ids = [ID for ID in eval_ids
                if os.path.exists(f"/data/cache_SAR/{ID}.npy") and os.path.exists(f"/data/cache_OPT/{ID}.npy")]
    N = len(eval_ids)
    print(f"{N} serie disponibili sar-opt", flush=True)
 
    try:
        encoderSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=attn).to(device)
        encoderOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=attn).to(device)
 
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
 
        w_path = project_work.get_artifact(f"moco_2D_weights_{job_name}").download("/data/eval_moco_weights.pth", overwrite=True)
        checkpoint = torch.load(w_path, map_location=device)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print("OK -> pesi MoCo caricato", flush=True)
    except Exception as e:
        print(f"EXC -> caricamento pesi MoCo: {e}", flush=True)
 
    eval_dataset = MoCo2encodersLoader(
        listIDs=eval_ids, sar_map={}, opt_map={}, transform=None,
        patch_size=patch_size,
        n_images1=n_images1, n_channels1=n_channels1,
        n_images2=n_images2, n_channels2=n_channels2,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
 
    all_q, all_k = [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for im_q, im_k in tqdm(eval_loader, mininterval=30.0):   # im_q = OPT, im_k = SAR
            im_q = im_q.to(device)
            im_k = im_k.to(device)
 
            _, q, k = model.contrastive_loss(im_q, im_k)
 
            all_q.append(q.float().cpu())
            all_k.append(k.float().cpu())
 
    Q = torch.cat(all_q).to(device)   
    K = torch.cat(all_k).to(device)    
 
 
    S = Q @ K.T
    diag = S.diag().unsqueeze(1)               
    
    rank_opt2sar = (S >= diag).sum(dim=1)        
 
    r = rank_opt2sar.float()
    results = {
        "job_name": job_name,
        "n_series": N,
        "sim_media_coppie_giuste": diag.mean().item(),
        "sim_media_tutte_le_coppie": S.mean().item(),
        "globale": {
            "top1": (r <= 1).float().mean().item(),
            "top5": (r <= 5).float().mean().item(),
            "top10": (r <= 10).float().mean().item(),
            "chance_top1": 1.0 / N,
            "chance_top5": 5.0 / N,
            "chance_top10": 10.0 / N,
            "chance_median_rank": (N + 1) / 2,
        },
    }
 
 
    id_pattern = re.compile(r"^(?P<event>.+)_(?P<zone>\d{1,2})[A-Z]{3}_x(?P<x>-?\d+)_y(?P<y>-?\d+)$")
    xs, ys, gids, group_map, n_unparsed = [], [], [], {}, 0
    ev_ids, event_map = [], {}            
    for ID in eval_ids:
        m = id_pattern.match(ID)
        if m is None:
            n_unparsed += 1
            xs.append(0.0)
            ys.append(0.0)
            gids.append(-n_unparsed)   
            ev_ids.append(-n_unparsed)
            continue
       
        key = f"{m.group('event')}_{m.group('zone')}"
        gids.append(group_map.setdefault(key, len(group_map)))
       
        ev_ids.append(event_map.setdefault(m.group('event'), len(event_map)))
        xs.append(float(m.group('x')))
        ys.append(float(m.group('y')))
 
    parsed_ok = n_unparsed <= N // 2
 
    ingroup = None
    if parsed_ok:
        gid_ev = torch.tensor(ev_ids, device=device)
        same = gid_ev[:, None] == gid_ev[None, :]           
        n_g = same.sum(dim=1)                               
        ok = n_g >= 2                                       
        rank_g = ((S >= diag) & same).sum(dim=1)[ok].float()   
        n = n_g[ok].float()
        ingroup = {
            "n_serie": ok.sum().item(),
            "dimensione_gruppo_mediana": n.median().item(),
            "dimensione_gruppo_max": n.max().item(),
            "top1": (rank_g <= 1).float().mean().item(),
            "top5": (rank_g <= 5).float().mean().item(),
            "top10": (rank_g <= 10).float().mean().item(),
        
            "chance_top1": (1.0 / n).mean().item(),
            "chance_top5": (torch.clamp(n, max=5) / n).mean().item(),
            "chance_top10": (torch.clamp(n, max=10) / n).mean().item(),
            "rank_normalizzato_mediano": ((rank_g - 1) / (n - 1)).median().item(),  
        }
        results["gruppo"] = ingroup
 
 
    g = results["globale"]
    W = 42
    print(f"mean cosine sim positive: {results['sim_media_coppie_giuste']:.4f} | mean cosine sim all: {results['sim_media_tutte_le_coppie']:.4f}", flush=True)
    print(f"{'':{W}s}{'top1':>9s}{'top5':>9s}{'top10':>9s}", flush=True)
    print(f"{f'global':{W}s}{100 * g['top1']:8.2f}%{100 * g['top5']:8.2f}%{100 * g['top10']:8.2f}%", flush=True)
    print(f"{'random':{W}s}{100 * g['chance_top1']:8.3f}%{100 * g['chance_top5']:8.3f}%{100 * g['chance_top10']:8.3f}%", flush=True)
    if ingroup is not None:
        label = f"local"
        print(f"{label:{W}s}{100 * ingroup['top1']:8.2f}%{100 * ingroup['top5']:8.2f}%{100 * ingroup['top10']:8.2f}%", flush=True)
        print(f"{'random':{W}s}{100 * ingroup['chance_top1']:8.3f}%{100 * ingroup['chance_top5']:8.3f}%{100 * ingroup['chance_top10']:8.3f}%", flush=True)
    print("=" * 69 + "\n", flush=True)
 
    if not parsed_ok:
        print("EXC -> troppi ID senza coordinate nel formato atteso, retrieval nel gruppo e analisi spaziale saltati", flush=True)
    else:
        x = torch.tensor(xs, device=device)
        y = torch.tensor(ys, device=device)
        gid = torch.tensor(gids, device=device)             
        valid = gid >= 0                                  
        idx = torch.arange(N, device=device)
 
        D = torch.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2) / tile_size_m
        D = torch.where(gid[:, None] == gid[None, :], D, torch.full_like(D, float('inf')))
 
        S_wrong = S.clone()
        S_wrong.fill_diagonal_(float('-inf'))
        d_opt2sar = D[idx, S_wrong.argmax(dim=1)][valid]    
 
        Dv = D[valid][:, valid]
        d_random = Dv[~torch.eye(Dv.shape[0], dtype=torch.bool, device=device)]
 
        spatial = {}
        print(f"tile = {tile_size_m:.0f} m | <=2 tile (~{2 * tile_size_m / 1000:.0f} km) | <=5 tile (~{5 * tile_size_m / 1000:.0f} km) | <=20 tile (~{20 * tile_size_m / 1000:.0f} km) | >20 tile", flush=True)
        print(f"{'':30s}{'adiacente':>11s}{'vicino':>9s}{'medio':>8s}{'lontano':>9s}{'altro ev.':>11s}{'mediana km':>12s}", flush=True)
        for key, label, d in [("opt2sar", "opt->sar", d_opt2sar),
                              ("random", "random", d_random)]:
            finite = torch.isfinite(d)
            spatial[key] = {
                "adiacente": (d <= 2).float().mean().item(),
                "vicino": ((d > 2) & (d <= 5)).float().mean().item(),
                "medio": ((d > 5) & (d <= 20)).float().mean().item(),
                "lontano": (finite & (d > 20)).float().mean().item(),
                "altro_evento": (~finite).float().mean().item(),
 
            }
            rr = spatial[key]
            print(f"{label:30s}{100 * rr['adiacente']:10.2f}%{100 * rr['vicino']:8.2f}%{100 * rr['medio']:7.2f}%{100 * rr['lontano']:8.2f}%{100 * rr['altro_evento']:10.2f}%", flush=True)
 
    return "TERMINATO -> retrieval MoCo 2D"


# notebook -> anomaly_detection

@handler()
def train_anomaly_detection(
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
def train_anomaly_detection_ocsvm(
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
    try:
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise RuntimeError("serve scikit-learn: aggiungilo ai requirements della function") from e
 
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







@handler()
def inspect_anomalies_tar(
    split: str = "test",
    n_show: int = 30,
):
    project_data = dh.get_project("datasets")
 
    print("Download artifact Floods_Anomalies...", flush=True)
    anom_path = project_data.get_artifact("Floods_Anomalies").download("/data/anomalies")
    print("OK -> download terminato", flush=True)
 
    print(f"\nContenuto di {anom_path}/{split}/:", flush=True)
    split_dir = os.path.join(anom_path, split)
    if os.path.isdir(split_dir):
        for f in sorted(os.listdir(split_dir)):
            full = os.path.join(split_dir, f)
            size_gb = os.path.getsize(full) / 1e9 if os.path.isfile(full) else None
            print(f"  {f}" + (f" ({size_gb:.2f} GB)" if size_gb is not None else " (cartella)"), flush=True)
    else:
        print(f"  ATTENZIONE: {split_dir} non esiste come cartella", flush=True)
        print(f"Contenuto di {anom_path}:", flush=True)
        for f in sorted(os.listdir(anom_path)):
            print(f"  {f}", flush=True)
 
    import tarfile
 
    for tar_name in ["S1RTC.tar", "S2L2A.tar"]:
        tar_path = os.path.join(split_dir, tar_name)
        print(f"\n{'=' * 60}", flush=True)
        print(f"{tar_name}: {tar_path}", flush=True)
        if not os.path.exists(tar_path):
            print("  ATTENZIONE: file non trovato a questo percorso", flush=True)
            continue
 
        try:
            tf = tarfile.open(tar_path, 'r:')
            names = tf.getnames()
            print(f"totale membri: {len(names)}", flush=True)
 
            print(f"\nprimi {n_show} nomi:", flush=True)
            for n in names[:n_show]:
                print(f"  {n}", flush=True)
 
            # primo segmento di ciascun path, per capire se c'e' un prefisso
            # di cartella inatteso (es. "data/...", "./...", niente prefisso, ecc.)
            top_level = sorted(set(n.split('/')[0] for n in names))
            print(f"\nprimi segmenti di path distinti (max 20 mostrati): {top_level[:20]}", flush=True)
 
            tf.close()
        except Exception as e:
            print(f"EXC -> apertura/lettura {tar_name}: {e}", flush=True)
 
    return "TERMINATO -> ispezione tar completata"








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






@handler()
def visualize_flood_overlay(
    dataset: str = "Floods_Anomalies_v2_crop_norm_full",   # artifact con SAR/, OPT/, OPT_RGB/, MASK/ (partNN.zip di .npy)
    n_samples: int = 10,
    min_flood_pixels: int = 1000,    # considera solo serie con almeno questi pixel classe flood (0 = nessun filtro)
    flood_class: int = 2,            # valore della classe flood nella maschera
    flood_instant: int = 3,          # istante (1-4) mostrato con la maschera nella seconda riga; gli altri restano bianchi
    stored_gamma: float = 2.2,       # rgb_gamma con cui e' stato costruito OPT_RGB: serve per tornare ai valori lineari
    display_gamma: float = 2.0,      # gamma in visualizzazione (>1 schiarisce i mezzi toni, 1.0 = lineare)
    display_stretch: bool = True,    # stira il contrasto (percentili 1-99) sui 4 istanti insieme
    display_saturation: float = 1.4, # 1.0 = colori come salvati, >1 piu' vivaci
    dpi: int = 150,                  # risoluzione dei PNG: 150 ~2 MB e ~1 s ciascuno, 100 ~1 MB e ~0.6 s (utile con molte serie)
    job_name: str = "flood_overlay",
    save_dir: str = "/data/flood_overlay",
):
    """
    Per ogni serie campionata salva un PNG 2x4 (solo ottico, da OPT_RGB):
      riga 0 = i 4 istanti in input (true color)
      riga 1 = solo l'istante `flood_instant` (default 3) con i pixel flood (MASK) in magenta,
               gli altri tre riquadri restano bianchi (ne' immagine ne' maschera).
    La maschera e' una sola per serie (annotazione dell'evento).
 
    I colori sono regolati solo in visualizzazione, senza ricostruire l'artifact: si torna ai
    valori lineari (stored_gamma), si stira il contrasto sui 4 istanti insieme, si applica
    display_gamma e si aumenta la saturazione. Stessa trasformazione per i tre canali e per
    tutti gli istanti, quindi i colori restano coerenti tra un istante e l'altro.
 
    OPT_RGB e MASK vengono letti direttamente dagli zip, una serie alla volta, senza estrarli su
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
 
    def _plot(rgb_arr, flood_mask, ID, n_flood, out_path):
        """rgb_arr: (3, T, H, W) in [0,1] gia' regolato per la visualizzazione; flood_mask: (H, W) bool."""
        try:
            T = rgb_arr.shape[1]
            t_flood = flood_instant - 1
            if not 0 <= t_flood < T:
                raise ValueError(f"flood_instant={flood_instant} fuori da 1..{T}")
 
            fig, axes = plt.subplots(2, T, figsize=(4 * T, 8), squeeze=False)
            pct = 100.0 * n_flood / flood_mask.size
            fig.suptitle(f"Serie: {ID} | flood: {n_flood} px ({pct:.1f}%), magenta = classe {flood_class}", fontsize=14)
 
            for t in range(T):
                rgb = np.clip(np.transpose(rgb_arr[:, t], (1, 2, 0)), 0.0, 1.0)     # (H, W, 3)
 
                axes[0, t].imshow(rgb)
                axes[0, t].set_title(f"Input t{t + 1}")
                axes[0, t].axis('off')
 
                if t == t_flood:
                    over = rgb.copy()
                    over[flood_mask] = [1.0, 0.0, 1.0]
                    axes[1, t].imshow(over)
                    axes[1, t].set_title(f"Input t{t + 1} + flood")
                axes[1, t].axis('off')          # gli altri riquadri restano bianchi
 
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
    # 1) ID presenti in OPT_RGB e in MASK, e pixel flood di ogni maschera
    # ------------------------------------------------------------------
    rgb_ids = set()
    for part_path in _list_parts("OPT_RGB"):
        with zipfile.ZipFile(part_path, 'r') as z:
            rgb_ids.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
    print(f"{len(rgb_ids)} serie OPT_RGB trovate", flush=True)
 
    flood_counts = {}
    for part_path in _list_parts("MASK"):
        with zipfile.ZipFile(part_path, 'r') as z:
            for n in z.namelist():
                if n.endswith('.npy'):
                    m = np.load(io.BytesIO(z.read(n)))
                    flood_counts[n[:-4]] = int((m == flood_class).sum())
    print(f"{len(flood_counts)} maschere trovate", flush=True)
 
    common = rgb_ids & set(flood_counts)
    candidates = sorted(ID for ID in common if flood_counts[ID] >= max(min_flood_pixels, 0))
    print(f"{len(common)} serie con OPT_RGB e maschera, {len(candidates)} con >= {min_flood_pixels} px flood", flush=True)
 
    if not candidates:
        raise RuntimeError("Nessuna serie candidata: abbassa min_flood_pixels o controlla che l'artifact "
                           "contenga OPT_RGB/ e MASK/ (va costruito con la nuova crop_norm_artifact_anomalies)")
 
    sampled = random.sample(candidates, min(n_samples, len(candidates)))
    print(f"ID campionati: {sampled}", flush=True)
 
    # ------------------------------------------------------------------
    # 2) maschere dei campionati in memoria; OPT_RGB letto direttamente
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
 
                if mask.shape != (H, W):
                    print(f"ATTENZIONE -> {ID}: maschera {mask.shape} diversa dall'immagine {(H, W)}, "
                          f"ridimensiono (nearest): controlla che siano davvero allineate", flush=True)
                    rows = (np.arange(H) * mask.shape[0]) // H
                    cols = (np.arange(W) * mask.shape[1]) // W
                    mask = mask[rows][:, cols]
 
                flood_mask = (mask == flood_class)
                n_flood = int(flood_mask.sum())
                ok = _plot(rgb_arr, flood_mask, ID, n_flood, os.path.join(save_dir, f"OPT_{ID}_flood.png"))
                done.add(ID)
                if not ok:
                    n_fail += 1
 
                if verbose:
                    print(f"{'OK' if ok else 'EXC'} -> {ID} | rgb {rgb_arr.shape} | flood px: {n_flood}", flush=True)
                elif len(done) % 100 == 0 or len(done) == len(sampled):
                    el = time.time() - t_start
                    print(f"{len(done)}/{len(sampled)} serie | {el / 60:.1f} min trascorsi | "
                          f"~{el / len(done) * (len(sampled) - len(done)) / 60:.1f} min rimanenti | "
                          f"{n_fail} errori", flush=True)
 
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







import subprocess

@handler()
def download_impactmesh_split(
    repo_id: str = "ibm-esa-geospatial/ImpactMesh-Flood",
    split: str = None,   # NUOVO: None = scarica l'intero dataset (train+val+test), non solo uno split
    output_name: str = "Floods_Anomalies_val",
    max_workers: int = 8,
    hf_token: str = ""
,   # dataset pubblico CC-BY 4.0, di norma non serve
):
    os.environ["HF_HUB_DISABLE_XET"] = "1"        # non basta da sola (bug noto, vedi sotto) ma la lascio comunque
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"  # default e' solo 10s, troppo poco per tar da svariati GB
 
    # NUOVO: HF_HUB_DISABLE_XET da sola e' inaffidabile su piu' versioni recenti di huggingface_hub
    # (bug noto e diffuso) - l'unico modo confermato di ricadere davvero sul percorso HTTP classico
    # e' disinstallare hf-xet, cosi' huggingface_hub non lo trova importabile e non puo' usarlo
    print("Disinstallo hf-xet per forzare il download HTTP classico...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "hf-xet", "hf_xet"],
        check=False,   # non deve far fallire il job se il pacchetto non c'e' proprio o il nome non combacia
    )
 
    from huggingface_hub import snapshot_download
 
    project_data = dh.get_project("datasets")
 
    local_dir = "/data/impactmesh_download"
    os.makedirs(local_dir, exist_ok=True)
 
    allow_patterns = [f"{split}/*"] if split is not None else None   # NUOVO: None -> nessun filtro, scarica tutto
    print(f"Download {repo_id}" + (f" (solo '{split}/*')" if split else " (INTERO dataset, tutti gli split)") + "...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=local_dir,
        max_workers=max_workers,
        token=hf_token,
    )
    print("OK -> download terminato", flush=True)
 
    check_dir = os.path.join(local_dir, split) if split is not None else local_dir   # NUOVO
    if not os.path.isdir(check_dir):
        print(f"EXC -> '{check_dir}' non trovata dopo il download, controlla il nome dello split", flush=True)
        return "TERMINATO -> contenuto non trovato"
 
    for f in sorted(os.listdir(check_dir)):   # NUOVO: gestisce sia file sia sottocartelle (train/val/test)
        full_path = os.path.join(check_dir, f)
        if os.path.isfile(full_path):
            size_gb = os.path.getsize(full_path) / 1e9
            print(f"  {f} ({size_gb:.2f} GB)", flush=True)
        else:
            print(f"  {f}/ (cartella)", flush=True)
 
    print(f"Upload come artifact '{output_name}'...", flush=True)
    project_data.log_artifact(
        name=output_name,
        kind='artifact',
        source=local_dir,   # con split=None, local_dir contiene train/val/test; con uno split, solo quella sottocartella
    )
    print(f"OK -> artifact '{output_name}' caricato", flush=True)
 
    shutil.rmtree(local_dir, ignore_errors=True)
    return "TERMINATO -> contenuto scaricato e caricato come artifact"







@handler()
def inspect_mask_structure(
    anomalies_artifact: str = "Floods_Anomalies",
    split: str = "test",
    n_samples: int = 5,
):
    project_data = dh.get_project("datasets")
 
    print(f"Download {anomalies_artifact}...", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/anomalies_mask_check")
    print("OK -> download terminato", flush=True)
 
    mask_tar_path = os.path.join(anom_path, split, "MASK.tar")
    if not os.path.exists(mask_tar_path):
        print(f"EXC -> {mask_tar_path} non trovato", flush=True)
        return "TERMINATO -> MASK.tar non trovato"
 
    tf = tarfile.open(mask_tar_path, 'r:')
    names = tf.getnames()
    print(f"\ntotale membri in MASK.tar ({split}): {len(names)}", flush=True)
    print(f"primi 10 nomi:", flush=True)
    for n in names[:10]:
        print(f"  {n}", flush=True)
 
    # stessi segmenti di path distinti che avevamo gia' controllato per S1RTC/S2L2A
    top_level = sorted(set(n.split('/')[0] for n in names))
    print(f"\nprimi segmenti di path distinti: {top_level[:10]}", flush=True)
 
    # apro i primi n_samples membri .zip (presumo stesso formato .zarr.zip, da confermare)
    sample_names = [n for n in names if n.endswith('.zip')][:n_samples]
    if not sample_names:
        print("\nEXC -> nessun membro .zip trovato - il formato potrebbe essere diverso da .zarr.zip", flush=True)
        print("primi 10 nomi SENZA filtro per vedere l'estensione reale:", flush=True)
        for n in names[:10]:
            print(f"  {n}", flush=True)
        tf.close()
        return "TERMINATO -> formato inatteso, vedi log"
 
    extract_tmp = "/data/mask_extract_tmp"
    for name in sample_names:
        print(f"\n{'=' * 60}", flush=True)
        print(f"ispeziono: {name}", flush=True)
        try:
            os.makedirs(extract_tmp, exist_ok=True)
            tf.extract(name, extract_tmp)
            local_path = os.path.join(extract_tmp, name)
 
            store = zarr.storage.ZipStore(local_path, mode='r')
            ds = xr.open_zarr(store, consolidated=True)
 
            print(f"variabili: {list(ds.data_vars)}", flush=True)
            print(f"coordinate: {list(ds.coords)}", flush=True)
            print(f"dimensioni: {dict(ds.dims)}", flush=True)
 
            for var in ds.data_vars:
                arr = ds[var].values
                print(f"  variabile '{var}': shape={arr.shape}, dtype={arr.dtype}", flush=True)
                vals, counts = np.unique(arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr, return_counts=True)
                print(f"    valori unici presenti: {dict(zip(vals.tolist(), counts.tolist()))}", flush=True)
 
            ds.close()
            store.close()
        except Exception as e:
            print(f"EXC -> {name}: {e}", flush=True)
        finally:
            import shutil
            shutil.rmtree(extract_tmp, ignore_errors=True)
 
    tf.close()
    return "TERMINATO -> ispezione MASK completata"


@handler()
def inspect_mask_tif(
    anomalies_artifact: str = "Floods_Anomalies_val",
    split: str = "val",
    n_samples: int = 5,
):
    project_data = dh.get_project("datasets")
 
    print(f"Download {anomalies_artifact}...", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/anomalies_mask_check")
    print("OK -> download terminato", flush=True)
 
    mask_tar_path = os.path.join(anom_path, split, "MASK.tar")
    if not os.path.exists(mask_tar_path):
        print(f"EXC -> {mask_tar_path} non trovato", flush=True)
        return "TERMINATO -> MASK.tar non trovato"
 
    tf = tarfile.open(mask_tar_path, 'r:')
    names = tf.getnames()
    print(f"\ntotale membri in MASK.tar ({split}): {len(names)}", flush=True)
 
    # controllo esplicito: esistono suffissi diversi da '_annotation_flood.tif'?
    # (es. un file separato '_annotation_water.tif' che non comparirebbe tra i primi 10)
    suffixes = set()
    for n in names:
        base = os.path.basename(n)
        if '_annotation_' in base:
            suffixes.add(base.split('_annotation_')[-1])
    print(f"suffissi '_annotation_*' distinti trovati in TUTTI i {len(names)} nomi: {suffixes}", flush=True)
 
    extract_tmp = "/data/mask_tif_extract_tmp"
    sample_names = names[:n_samples]
 
    for name in sample_names:
        print(f"\n{'=' * 60}", flush=True)
        print(f"ispeziono: {name}", flush=True)
        try:
            os.makedirs(extract_tmp, exist_ok=True)
            tf.extract(name, extract_tmp)
            local_path = os.path.join(extract_tmp, name)
 
            with rasterio.open(local_path) as src:
                print(f"  count bande: {src.count}, dtype: {src.dtypes}, shape: {src.shape} (H,W)", flush=True)
                print(f"  CRS: {src.crs}", flush=True)
                print(f"  nodata dichiarato: {src.nodata}", flush=True)
                data = src.read()   # (bande, H, W)
                print(f"  shape array letto: {data.shape}", flush=True)
 
                vals, counts = np.unique(data, return_counts=True)
                print(f"  valori unici presenti: {dict(zip(vals.tolist(), counts.tolist()))}", flush=True)
        except Exception as e:
            print(f"EXC -> {name}: {e}", flush=True)
        finally:
            shutil.rmtree(extract_tmp, ignore_errors=True)
 
    tf.close()
    return "TERMINATO -> ispezione MASK (.tif) completata"



@handler()
def inspect_full_metadata(
    anomalies_artifact: str = "Floods_Anomalies_val",
    split: str = "val",
    n_samples: int = 2,
):
    project_data = dh.get_project("datasets")
 
    print(f"Download {anomalies_artifact}...", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/full_metadata_check")
    print("OK -> download terminato\n", flush=True)
 
    split_dir = os.path.join(anom_path, split)
    extract_tmp = "/data/full_metadata_extract_tmp"
 
    for modality, suffix in [("S1RTC", "S1RTC"), ("S2L2A", "S2L2A")]:
        tar_paths = sorted(set(
            glob_module.glob(os.path.join(split_dir, f"{modality}.tar")) +
            glob_module.glob(os.path.join(split_dir, f"{modality}_*.tar"))
        ))
        if not tar_paths:
            print(f"EXC -> nessun tar trovato per {modality}", flush=True)
            continue
 
        pattern = re.compile(r"^[^/]+/(.*?)_" + re.escape(suffix) + r"\.zarr\.zip$")
 
        for tar_path in tar_paths:
            tf = tarfile.open(tar_path, 'r:')
            sample_members = [n for n in tf.getnames() if pattern.match(n)][:n_samples]
 
            for name in sample_members:
                print(f"\n{'=' * 70}", flush=True)
                print(f"{modality} - {name}", flush=True)
                try:
                    os.makedirs(extract_tmp, exist_ok=True)
                    tf.extract(name, extract_tmp)
                    local_path = os.path.join(extract_tmp, name)
 
                    store = zarr.storage.ZipStore(local_path, mode='r')
                    ds = xr.open_zarr(store, consolidated=True)
 
                    print("ATTRIBUTI GLOBALI DEL DATASET (ds.attrs):", flush=True)
                    for k, v in ds.attrs.items():
                        print(f"  {k} = {v}", flush=True)
 
                    print("\nCOORDINATE (ds.coords):", flush=True)
                    for c in ds.coords:
                        vals = ds.coords[c].values
                        preview = vals if vals.size <= 10 else f"{vals[:5]} ... ({vals.size} valori totali)"
                        print(f"  {c}: {preview}", flush=True)
 
                    print("\nATTRIBUTI PER VARIABILE:", flush=True)
                    for var in ds.data_vars:
                        if ds[var].attrs:
                            print(f"  {var}: {dict(ds[var].attrs)}", flush=True)
 
                    ds.close()
                    store.close()
                except Exception as e:
                    print(f"EXC -> {name}: {e}", flush=True)
                finally:
                    shutil.rmtree(extract_tmp, ignore_errors=True)
 
            tf.close()
            break   # un solo file tar per modalita' e' sufficiente per questa ispezione
 
    return "TERMINATO -> ispezione metadati completa"


@handler()
def extract_mask_pixel_counts(
    input_csv_artifact: str = "anomaly_ds_filtrato",   # il CSV che hai gia', con Nome_Serie e le altre colonne
    anomalies_artifact: str = "Floods_Anomalies",       # o Floods_Anomalies_v2 se nel frattempo e' pronto
    split: list = None,                                # None -> ["train", "val", "test"]
    output_name: str = "anomaly_ds_with_mask",
):
    project_data = dh.get_project("datasets")

    if split is None:
        split = ["train", "val", "test"]

    print(f"Lettura CSV di input: {input_csv_artifact}", flush=True)
    csv_path = project_data.get_artifact(input_csv_artifact).download("/data/input_mask_extract.csv", overwrite=True)
    df = pd.read_csv(csv_path)
    print(f"OK -> {len(df)} righe", flush=True)

    print(f"Download {anomalies_artifact}...", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/anomalies_mask_extract")
    print("OK -> download terminato", flush=True)

    # ------------------------------------------------------------------
    # indicizzo MASK.tar per ogni split - stesso schema 'cerca singolo o
    # diviso in parti' gia' usato per S1RTC/S2L2A, per sicurezza/uniformita'
    # anche se qui non ci aspettiamo file divisi
    # ------------------------------------------------------------------
    def _list_mask_members(split_dir):
        tar_paths = sorted(set(
            glob_module.glob(os.path.join(split_dir, "MASK.tar")) +
            glob_module.glob(os.path.join(split_dir, "MASK_*.tar"))
        ))
        handles = []
        members = {}   # ID -> (tar_handle, nome membro)
        for tar_path in tar_paths:
            tf = tarfile.open(tar_path, 'r:')
            handles.append(tf)
            for name in tf.getnames():
                base = os.path.basename(name)
                if base.endswith('_annotation_flood.tif'):
                    ID = base[:-len('_annotation_flood.tif')]
                    members[ID] = (tf, name)
        return handles, members

    all_handles = []
    id_to_member = {}
    for split in split:
        split_dir = os.path.join(anom_path, split)
        if not os.path.isdir(split_dir):
            print(f"EXC -> split '{split}' non trovato in {anom_path}, salto", flush=True)
            continue
        handles, members = _list_mask_members(split_dir)
        all_handles.extend(handles)
        print(f"OK -> '{split}': {len(members)} maschere indicizzate", flush=True)
        for ID, v in members.items():
            if ID not in id_to_member:   # non dovrebbero sovrapporsi tra split, ma il primo trovato vince
                id_to_member[ID] = v

    print(f"OK -> {len(id_to_member)} ID totali indicizzati su tutti gli split\n", flush=True)

    # ------------------------------------------------------------------
    # per ogni riga del CSV, estraggo e leggo la maschera corrispondente
    # ------------------------------------------------------------------
    mask_water, mask_flood = [], []
    n_ok, n_missing, n_failed = 0, 0, 0
    extract_tmp = "/data/mask_extract_tmp"
    total = len(df)

    for i, ID in enumerate(df['Nome_Serie']):
        if i % 1000 == 0:
            print(f"  {i}/{total} ({n_ok} ok, {n_missing} non trovate, {n_failed} errori)", flush=True)

        if ID not in id_to_member:
            mask_water.append(np.nan)
            mask_flood.append(np.nan)
            n_missing += 1
            continue

        tf, member_name = id_to_member[ID]
        try:
            os.makedirs(extract_tmp, exist_ok=True)
            tf.extract(member_name, extract_tmp)
            local_path = os.path.join(extract_tmp, member_name)
            with rasterio.open(local_path) as src:
                data = src.read(1)
            mask_water.append(int((data == 1).sum()))
            mask_flood.append(int((data == 2).sum()))
            n_ok += 1
        except Exception as e:
            print(f"EXC -> {ID}: {e}", flush=True)
            mask_water.append(np.nan)
            mask_flood.append(np.nan)
            n_failed += 1
        finally:
            shutil.rmtree(extract_tmp, ignore_errors=True)

    for tf in all_handles:
        tf.close()

    df['Mask_water'] = mask_water
    df['Mask_flood'] = mask_flood

    print(f"\nOK -> {n_ok} maschere lette, {n_missing} ID non trovati, {n_failed} errori di lettura", flush=True)

    out_path = f"/data/{output_name}.csv"
    df.to_csv(out_path, index=False)

    project_data.log_artifact(name=output_name, kind='artifact', source=out_path)
    print(f"OK -> '{output_name}' caricato come artifact", flush=True)

    return f"TERMINATO -> {n_ok} maschere estratte, {n_missing} mancanti, {n_failed} errori"    


import os
import re
import shutil
import tarfile

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import zarr


@handler()
def build_full_metadata_csv(
    anomalies_artifact: str = "Floods_Anomalies_v2",
    splits: list = None,            # None -> ["train", "val", "test"]
    output_name: str = "anomaly_ds_v2",
    max_series: int = 0,            # 0 = tutte; usa un numero piccolo per un primo test
):
    project_data = dh.get_project("datasets")

    if splits is None:
        splits = ["train", "val", "test"]

    print(f"Download {anomalies_artifact}...", flush=True)
    anom_path = project_data.get_artifact(anomalies_artifact).download("/data/full_meta_dl")
    print("OK -> download terminato\n", flush=True)

    # ------------------------------------------------------------------
    # indicizzazione tar - stesso schema 'singolo o diviso in parti'
    # gia' usato per crop_norm_artifact_anomalies / extract_mask_pixel_counts
    # ------------------------------------------------------------------
    def _list_zarr_groups(split_dir, modality_label):
        tar_paths = sorted(set(
            glob.glob(os.path.join(split_dir, f"{modality_label}.tar")) +
            glob.glob(os.path.join(split_dir, f"{modality_label}_*.tar"))
        ))
        pattern = re.compile(r"^[^/]+/(.*?)_" + re.escape(modality_label) + r"\.zarr\.zip$")
        handles = []
        groups = {}   # ID -> (tar_handle, nome membro) - un solo file per ID, confermato in precedenza
        for tar_path in tar_paths:
            tf = tarfile.open(tar_path, 'r:')
            handles.append(tf)
            for name in tf.getnames():
                m = pattern.match(name)
                if m:
                    ID = m.group(1)
                    if ID not in groups:
                        groups[ID] = (tf, name)
        return handles, groups

    def _list_mask_members(split_dir):
        tar_paths = sorted(set(
            glob_module.glob(os.path.join(split_dir, "MASK.tar")) +
            glob_module.glob(os.path.join(split_dir, "MASK_*.tar"))
        ))
        handles = []
        members = {}
        for tar_path in tar_paths:
            tf = tarfile.open(tar_path, 'r:')
            handles.append(tf)
            for name in tf.getnames():
                base = os.path.basename(name)
                if base.endswith('_annotation_flood.tif'):
                    ID = base[:-len('_annotation_flood.tif')]
                    members[ID] = (tf, name)
        return handles, members

    rows = []
    extract_tmp = "/data/full_meta_extract_tmp"
    n_ok, n_failed = 0, 0
    stop = False

    for split in splits:
        if stop:
            break

        split_dir = os.path.join(anom_path, split)
        if not os.path.isdir(split_dir):
            print(f"EXC -> split '{split}' non trovato, salto", flush=True)
            continue

        print(f"Indicizzazione split '{split}'...", flush=True)
        handles_s1, groups_s1 = _list_zarr_groups(split_dir, "S1RTC")
        handles_s2, groups_s2 = _list_zarr_groups(split_dir, "S2L2A")
        handles_mask, members_mask = _list_mask_members(split_dir)

        common_ids = sorted(set(groups_s1) & set(groups_s2))
        print(f"OK -> '{split}': {len(common_ids)} serie con SAR+OTTICO, "
              f"{len(members_mask)} maschere disponibili", flush=True)

        for ID in common_ids:
            if max_series > 0 and n_ok >= max_series:
                stop = True
                break

            try:
                os.makedirs(extract_tmp, exist_ok=True)

                tf_s1, name_s1 = groups_s1[ID]
                tf_s1.extract(name_s1, extract_tmp)
                store_s1 = zarr.storage.ZipStore(os.path.join(extract_tmp, name_s1), mode='r')
                ds_s1 = xr.open_zarr(store_s1, consolidated=True)
                attrs_s1 = ds_s1.attrs

                tf_s2, name_s2 = groups_s2[ID]
                tf_s2.extract(name_s2, extract_tmp)
                store_s2 = zarr.storage.ZipStore(os.path.join(extract_tmp, name_s2), mode='r')
                ds_s2 = xr.open_zarr(store_s2, consolidated=True)
                attrs_s2 = ds_s2.attrs

                data_sar = attrs_s1['datetime'].split(';')
                bbox_sar = attrs_s1['proj_bbox'].split(';')
                epsg_sar = attrs_s1['proj_code'].split(';')[0]

                data_opt = attrs_s2['datetime'].split(';')
                cloud_opt = attrs_s2['eo_cloud_cover'].split(';')
                mgrs_opt = attrs_s2['s2_mgrs_tile'].split(';')[0]
                epsg_opt = attrs_s2['proj_code'].split(';')[0]

                row = {
                    'Nome_Serie': ID,
                    'Data_1_org_SAR': data_sar[0], 'Data_2_org_SAR': data_sar[1],
                    'Data_3_org_SAR': data_sar[2], 'Data_4_org_SAR': data_sar[3],
                    'Bbox_1_SAR': bbox_sar[0], 'Bbox_2_SAR': bbox_sar[1],
                    'Bbox_3_SAR': bbox_sar[2], 'Bbox_4_SAR': bbox_sar[3],
                    'Lat_Centro_SAR': float(attrs_s1['center_lat']),
                    'Lon_Centro_SAR': float(attrs_s1['center_lon']),
                    'EPSG_SAR': epsg_sar,
                    'Data_1_org_OPT': data_opt[0], 'Data_2_org_OPT': data_opt[1],
                    'Data_3_org_OPT': data_opt[2], 'Data_4_org_OPT': data_opt[3],
                    'Cloud_1_OPT': float(cloud_opt[0]), 'Cloud_2_OPT': float(cloud_opt[1]),
                    'Cloud_3_OPT': float(cloud_opt[2]), 'Cloud_4_OPT': float(cloud_opt[3]),
                    'Lat_Centro_OPT': float(attrs_s2['center_lat']),
                    'Lon_Centro_OPT': float(attrs_s2['center_lon']),
                    'MGRS_Tile_OPT': mgrs_opt,
                    'EPSG_OPT': epsg_opt,
                }

                ds_s1.close(); store_s1.close()
                ds_s2.close(); store_s2.close()

                if ID in members_mask:
                    tf_m, name_m = members_mask[ID]
                    tf_m.extract(name_m, extract_tmp)
                    with rasterio.open(os.path.join(extract_tmp, name_m)) as src:
                        mdata = src.read(1)
                    row['water_mask'] = int((mdata == 1).sum())
                    row['flood_mask'] = int((mdata == 2).sum())
                else:
                    row['water_mask'] = np.nan
                    row['flood_mask'] = np.nan

                rows.append(row)
                n_ok += 1
                if n_ok % 500 == 0:
                    print(f"  {n_ok} serie processate...", flush=True)

            except Exception as e:
                n_failed += 1
                print(f"EXC -> serie {ID} ({split}): {e}", flush=True)
            finally:
                shutil.rmtree(extract_tmp, ignore_errors=True)

        for tf in handles_s1 + handles_s2 + handles_mask:
            tf.close()

    print(f"\nOK -> {n_ok} serie processate, {n_failed} fallite", flush=True)

    if n_ok == 0:
        return "TERMINATO -> nessuna serie processata, controlla i log EXC sopra"

    df = pd.DataFrame(rows)
    out_path = f"/data/{output_name}.csv"
    df.to_csv(out_path, index=False)

    project_data.log_artifact(name=output_name, kind='artifact', source=out_path)
    print(f"OK -> '{output_name}' caricato come artifact ({len(df)} righe)", flush=True)

    return f"TERMINATO -> {n_ok} serie estratte, {n_failed} fallite"






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
from scipy.stats import rankdata, spearmanr

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
def analyze_anomaly_embeddings(
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
    







 









    