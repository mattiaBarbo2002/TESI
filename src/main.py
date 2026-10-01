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
import tarfile
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from digitalhub_runtime_python import handler

from multimodal_3Dconv_attention import Singlemodal_CAE
from multimodal_3Dconv_attention import Singlemodal_CAE_2d
from moco.loader import Singlemodal_Loader
from moco.loader import MoCo2encodersLoader
from moco.builder import MoCo2encoders
from moco.builder import MoCo2encoders_2d

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
    n_threshold_series: int = 2000,                 # serie normali per calcolo soglia
    num_anomalies: int = 100,
    n_std: float = 1.0,                             # quante +- deviazioni standard
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
    normal_thr_ids = random.sample(train_ids, min(n_threshold_series, len(train_ids)))
    normal_test_ids = random.sample(test_ids, len(test_ids))

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

    normal_thr_ids = [ID for ID in normal_thr_ids if ID in normal_ids]
    normal_test_ids = [ID for ID in normal_test_ids if ID in normal_ids]
    print(f"OK -> {len(normal_thr_ids)} serie soglia, {len(normal_test_ids)} serie test normali", flush=True)

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





# EXTRAXT ANOMALIES

def _list_zarr_groups(tar_path, suffix):
    """
    Apre il tar e raggruppa i membri per ID di serie, senza estrarre nulla.
    Formato reale: "{SUFFIX}/{ID}_{SUFFIX}.zarr.zip" - un solo file per serie
    (uno ZipStore Zarr, non una cartella con tanti membri).
    Ritorna (tar_handle_aperto, {ID: [nome membro nel tar]}).
    Il tar_handle va chiuso dal chiamante quando non serve piu'.
    """
    tf = tarfile.open(tar_path, 'r:')
    pattern = re.compile(r"^[^/]+/(.*?)_" + re.escape(suffix) + r"\.zarr\.zip$")
    groups = {}
    for member in tf.getmembers():
        m = pattern.match(member.name)
        if m:
            groups.setdefault(m.group(1), []).append(member.name)
    return tf, groups
 
 
def _load_and_process_zarr(zip_path, band_names_wanted, band_name_map=None):
    """
    Apre un file .zarr.zip (ZipStore Zarr) con variabile 'bands' (time,band,y,x)
    e coordinata 'band' con i nomi veri delle bande, seleziona i canali richiesti
    PER NOME (non per posizione), azzera i pixel non validi secondo nan_mask,
    normalizza a percentile 2-98 su tutta la serie (canali+istanti insieme,
    stessa logica di Singlemodal_Loader._load_and_process), e ritorna un array
    (n_channels, n_images, H, W) float32 in [0,1].
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
 
    # anche i pixel gia' marcati come fill_value (es. -9999 per l'ottico) finiscono
    # fuori dal range plausibile: li tratto come mancanti allo stesso modo dei NaN
    data = np.where(data < -1000, np.nan, data)
 
    p_low = np.nanpercentile(data, 2)
    p_high = np.nanpercentile(data, 98)
    scale = p_high - p_low if (p_high - p_low) > 1e-6 else 1.0
    data = (data - p_low) / scale
    data = np.nan_to_num(data, nan=0.0)          # mancante -> 0, stessa convenzione del padding nel resto del progetto
    data = np.clip(data, 0.0, 1.0).astype(np.float32)
 
    return data
 
 
@handler()
def build_precomputed_cache_anomalies(
    split: str = "test",          # "train" | "val" | "test", cartella dentro Floods_Anomalies
    n_series_sample: int = 5,     # NOTA: parti piccolo (5-10) per verificare che funzioni, poi alza a 5000
    seed: int = 0,
    max_part_gb: float = 15.0,
):
    project_data = dh.get_project("datasets")
 
    sar_bands = ["vv", "vh"]                                                    # n_channels1 = 2
    opt_bands_project = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]  # n_channels2 = 10
    opt_band_map = {b: ("B0" + b[1:] if len(b) == 2 and b[1].isdigit() else b) for b in opt_bands_project}
 
    print(f"Download artifact Floods_Anomalies...", flush=True)
    anom_path = project_data.get_artifact("Floods_Anomalies").download("/data/anomalies")
    print("OK -> download terminato", flush=True)
 
    s1_tar_path = os.path.join(anom_path, split, "S1RTC.tar")
    s2_tar_path = os.path.join(anom_path, split, "S2L2A.tar")
 
    print(f"Indicizzazione S1RTC ({split})...", flush=True)
    tf_s1, groups_s1 = _list_zarr_groups(s1_tar_path, "S1RTC")
    print(f"OK -> {len(groups_s1)} serie SAR trovate", flush=True)
 
    print(f"Indicizzazione S2L2A ({split})...", flush=True)
    tf_s2, groups_s2 = _list_zarr_groups(s2_tar_path, "S2L2A")
    print(f"OK -> {len(groups_s2)} serie OTTICO trovate", flush=True)
 
    common_ids = sorted(set(groups_s1) & set(groups_s2))
    print(f"{len(common_ids)} serie complete SAR+OTTICO", flush=True)
 
    random.seed(seed)
    sample_ids = random.sample(common_ids, min(n_series_sample, len(common_ids)))
    print(f"campionate {len(sample_ids)} serie (seed={seed})", flush=True)
 
    local_root = "/data/anomalies_cache_upload"
    sar_dir = os.path.join(local_root, "SAR")
    opt_dir = os.path.join(local_root, "OPT")
    os.makedirs(sar_dir, exist_ok=True)
    os.makedirs(opt_dir, exist_ok=True)
 
    extract_tmp = "/data/anomalies_extract_tmp"
    n_ok, n_failed = 0, 0
 
    for ID in sample_ids:
        try:
            os.makedirs(extract_tmp, exist_ok=True)
            for name in groups_s1[ID]:
                tf_s1.extract(name, extract_tmp)
            for name in groups_s2[ID]:
                tf_s2.extract(name, extract_tmp)
 
            s1_group_path = os.path.join(extract_tmp, "S1RTC", f"{ID}_S1RTC.zarr.zip")
            s2_group_path = os.path.join(extract_tmp, "S2L2A", f"{ID}_S2L2A.zarr.zip")
 
            im_sar = _load_and_process_zarr(s1_group_path, sar_bands)
            im_opt = _load_and_process_zarr(s2_group_path, opt_bands_project, band_name_map=opt_band_map)
 
            np.save(os.path.join(sar_dir, f"{ID}.npy"), im_sar)
            np.save(os.path.join(opt_dir, f"{ID}.npy"), im_opt)
            n_ok += 1
            if n_ok <= 3:
                print(f"OK -> {ID}: SAR {im_sar.shape} min/max {im_sar.min():.3f}/{im_sar.max():.3f}, "
                      f"OPT {im_opt.shape} min/max {im_opt.min():.3f}/{im_opt.max():.3f}", flush=True)
        except Exception as e:
            n_failed += 1
            print(f"EXC -> serie {ID}: {e}", flush=True)
        finally:
            shutil.rmtree(extract_tmp, ignore_errors=True)
 
    tf_s1.close()
    tf_s2.close()
    print(f"OK -> {n_ok} serie processate, {n_failed} fallite", flush=True)
 
    if n_ok == 0:
        return "TERMINATO -> nessuna serie processata, controlla i log EXC sopra"
 
    # ------------------------------------------------------------------
    # scrivo le due cartelle in parti sotto max_part_gb (streaming: scrivo
    # ed elimino ogni .npy dopo averlo aggiunto allo zip), poi le carico
    # come UN SOLO artifact con sottocartelle SAR/OPT - stesso schema di
    # build_precomputed_cache.py per il dataset "normale"
    # ------------------------------------------------------------------
    def _write_parts(src_dir, dest_dir, max_part_bytes):
        part_num, current_zip, current_path, current_size = 1, None, None, 0
 
        def _open():
            nonlocal current_zip, current_path, current_size
            current_path = os.path.join(dest_dir, f"part{part_num:02d}.zip")
            current_zip = zipfile.ZipFile(current_path, 'w', zipfile.ZIP_DEFLATED)
            current_size = 0
 
        def _close():
            nonlocal current_zip, part_num
            current_zip.close()
            print(f"OK -> {dest_dir} parte {part_num} pronta ({os.path.getsize(current_path) / 1e9:.2f} GB)", flush=True)
            part_num += 1
 
        _open()
        for fname in os.listdir(src_dir):
            full_path = os.path.join(src_dir, fname)
            fsize = os.path.getsize(full_path)
            if current_size > 0 and current_size + fsize > max_part_bytes:
                _close()
                _open()
            current_zip.write(full_path, arcname=fname)
            current_size += fsize
            os.remove(full_path)
        _close()
        shutil.rmtree(src_dir, ignore_errors=True)
 
    max_part_bytes = int(max_part_gb * 1024 ** 3)
    local_upload_root = "/data/anomalies_cache_parts"
    os.makedirs(os.path.join(local_upload_root, "SAR"), exist_ok=True)
    os.makedirs(os.path.join(local_upload_root, "OPT"), exist_ok=True)
 
    _write_parts(sar_dir, os.path.join(local_upload_root, "SAR"), max_part_bytes)
    _write_parts(opt_dir, os.path.join(local_upload_root, "OPT"), max_part_bytes)
 
    try:
        project_data.log_artifact(
            name=f"Floods_Anomalies_{split}_crop_norm_100",
            kind='artifact',
            source=local_upload_root,
        )
        print(f"OK -> artifact Floods_Anomalies_{split}_crop_norm_100 caricato", flush=True)
        shutil.rmtree(local_upload_root, ignore_errors=True)
    except Exception as e:
        print(f"EXC -> upload artifact: {e}", flush=True)
 
    return "TERMINATO -> cache anomalie precalcolata"



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
 
    # indicizzo gli ID gia' pronti nella cache (senza estrarre), poi estraggo solo i campioni scelti
    try:
        if test_sar:
            sar_available = set()
            for part_path in sorted(glob(os.path.join(dataset_path, "SAR", "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    sar_available.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
            random_sar_ids = random.sample(sorted(sar_available), min(n_samples, len(sar_available))) if sar_available else []
            print(f"{len(sar_available)} serie SAR trovate", flush=True)
 
            os.makedirs(cache_sar_dir, exist_ok=True)
            wanted_sar = {f"{ID}.npy" for ID in random_sar_ids}
            for part_path in sorted(glob(os.path.join(dataset_path, "SAR", "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    for name in z.namelist():
                        if name in wanted_sar:
                            z.extract(name, cache_sar_dir)
 
        if test_opt:
            opt_available = set()
            for part_path in sorted(glob(os.path.join(dataset_path, "OPT", "part*.zip"))):
                with zipfile.ZipFile(part_path, 'r') as z:
                    opt_available.update(n[:-4] for n in z.namelist() if n.endswith('.npy'))
            random_opt_ids = random.sample(sorted(opt_available), min(n_samples, len(opt_available))) if opt_available else []
            print(f"{len(opt_available)} serie OPT trovate", flush=True)
 
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

 









    