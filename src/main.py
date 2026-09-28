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
import re
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
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    max_series: int = 0,                 
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
        print(f"OK -> {len(eval_ids)} serie held-out caricate", flush=True)
    except Exception as e:
        print(f"EXC -> caricamento lista held-out: {e}", flush=True)
        return "TERMINATO -> lista held-out non trovata"
 
    if max_series > 0 and len(eval_ids) > max_series:
        random.seed(0)
        eval_ids = random.sample(eval_ids, max_series)
        print(f"sottocampionate a {len(eval_ids)} serie", flush=True)
 
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
        print(f"OK -> {modality}: serie held-out estratte da {len(part_files)} parti", flush=True)
 
    eval_ids = [ID for ID in eval_ids
                if os.path.exists(f"/data/cache_SAR/{ID}.npy") and os.path.exists(f"/data/cache_OPT/{ID}.npy")]
    N = len(eval_ids)
    print(f"{N} serie disponibili in entrambe le modalita'", flush=True)
    if N < 10:
        return "TERMINATO -> troppe poche serie per il retrieval"
 
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
            attn=attn
        ).to(device)
 
        w_path = project_work.get_artifact(f"moco_2D_weights_{job_name}").download("/data/eval_moco_weights.pth", overwrite=True)
        checkpoint = torch.load(w_path, map_location=device)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print("OK -> modello MoCo caricato", flush=True)
    except Exception as e:
        print(f"EXC -> caricamento modello MoCo: {e}", flush=True)
        return "TERMINATO -> modello MoCo non caricato"
 
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
 
            q = model.encoder_q(im_q)
            if simple_proj:
                q = model.pool_q(q).flatten(1)
            else:
                q = model.conv_q(q)
                if attn:
                    q_scores = model.attn_q(q).flatten(2).softmax(dim=-1)
                    q = torch.einsum('bcn,bkn->bck', q.flatten(2), q_scores).squeeze(-1)
                else:
                    q = model.pool_q(q).flatten(1)
            q = nn.functional.normalize(model.proj_q(q).float(), dim=1)
 
            k = model.encoder_k(im_k)
            if simple_proj:
                k = model.pool_k(k).flatten(1)
            else:
                k = model.conv_k(k)
                if attn:
                    k_scores = model.attn_k(k).flatten(2).softmax(dim=-1)
                    k = torch.einsum('bcn,bkn->bck', k.flatten(2), k_scores).squeeze(-1)
                else:
                    k = model.pool_k(k).flatten(1)
            k = nn.functional.normalize(model.proj_k(k).float(), dim=1)
 
            all_q.append(q.cpu())
            all_k.append(k.cpu())
 
    Q = torch.cat(all_q).to(device)    
    K = torch.cat(all_k).to(device)   
 
    S = Q @ K.T
    diag = S.diag().unsqueeze(1)                 
 
    rank_opt2sar = (S >= diag).sum(dim=1)        
    rank_sar2opt = (S >= diag.T).sum(dim=0)      
 
    results = {
        "job_name": job_name,
        "n_series": N,
        "sim_media_coppie_giuste": diag.mean().item(),
        "sim_media_tutte_le_coppie": S.mean().item(),
        "chance_top1": 1.0 / N,
        "chance_top5": 5.0 / N,
        "chance_top10": 10.0 / N,
        "chance_median_rank": (N + 1) / 2,
    }
 
    print("\n" + "=" * 50, flush=True)
    print(f"RETRIEVAL su {N} serie held-out", flush=True)
    print(f"sim media coppie giuste: {results['sim_media_coppie_giuste']:.4f} | sim media tutte le coppie: {results['sim_media_tutte_le_coppie']:.4f}", flush=True)
    print(f"caso random -> top1: {100 * results['chance_top1']:.3f}% | top5: {100 * results['chance_top5']:.3f}% | top10: {100 * results['chance_top10']:.3f}% | median rank: {results['chance_median_rank']:.0f}", flush=True)
 
    for name, rank in [("OPT->SAR", rank_opt2sar), ("SAR->OPT", rank_sar2opt)]:
        r = rank.float()
        results[name] = {
            "top1": (r <= 1).float().mean().item(),
            "top5": (r <= 5).float().mean().item(),
            "top10": (r <= 10).float().mean().item(),
            "median_rank": r.median().item(),
        }
        print(f"{name} -> top1: {100 * results[name]['top1']:.2f}% | top5: {100 * results[name]['top5']:.2f}% | top10: {100 * results[name]['top10']:.2f}% | median rank: {results[name]['median_rank']:.0f}", flush=True)
    print("=" * 50 + "\n", flush=True)
 
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
 
    print(f"ANALISI SPAZIALE: {N - n_unparsed}/{N} ID con coordinate, {len(event_map)} eventi (attivazione+AOI), {len(group_map)} gruppi evento/zona", flush=True)
 
    if n_unparsed > N // 2:
        print("EXC -> troppi ID senza coordinate nel formato atteso, analisi spaziale saltata", flush=True)
    else:
        x = torch.tensor(xs, device=device)
        y = torch.tensor(ys, device=device)
        gid = torch.tensor(gids, device=device)             
        gid_ev = torch.tensor(ev_ids, device=device)        
        valid = gid >= 0                                    
        idx = torch.arange(N, device=device)
 
        
        D = torch.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2) / tile_size_m
        D = torch.where(gid[:, None] == gid[None, :], D, torch.full_like(D, float('inf')))
 
        
        S_wrong = S.clone()
        S_wrong.fill_diagonal_(float('-inf'))
        d_opt2sar = D[idx, S_wrong.argmax(dim=1)][valid]    
        d_sar2opt = D[S_wrong.argmax(dim=0), idx][valid]    
 
        
        Dv = D[valid][:, valid]
        d_random = Dv[~torch.eye(Dv.shape[0], dtype=torch.bool, device=device)]
 
        spatial = {}
        print(f"tile = {tile_size_m:.0f} m | adiacente <=1.5 tile | vicino <=5 tile (~{5 * tile_size_m / 1000:.0f} km) | medio <=20 tile (~{20 * tile_size_m / 1000:.0f} km) | lontano >20 tile", flush=True)
        print(f"{'':30s}{'adiacente':>11s}{'vicino':>9s}{'medio':>8s}{'lontano':>9s}{'altro ev.':>11s}{'mediana km':>12s}", flush=True)
        for key, label, d in [("opt2sar", "OPT->SAR (miglior errore)", d_opt2sar),
                              ("sar2opt", "SAR->OPT (miglior errore)", d_sar2opt),
                              ("caso", "caso (coppie a caso)", d_random)]:
            finite = torch.isfinite(d)
            spatial[key] = {
                "adiacente": (d <= 1.5).float().mean().item(),
                "vicino": ((d > 1.5) & (d <= 5)).float().mean().item(),
                "medio": ((d > 5) & (d <= 20)).float().mean().item(),
                "lontano": (finite & (d > 20)).float().mean().item(),
                "altro_evento": (~finite).float().mean().item(),
                "mediana_km": (d[finite].median() * tile_size_m / 1000).item() if finite.any() else float('nan'),
            }
            r = spatial[key]
            print(f"{label:30s}{100 * r['adiacente']:10.2f}%{100 * r['vicino']:8.2f}%{100 * r['medio']:7.2f}%{100 * r['lontano']:8.2f}%{100 * r['altro_evento']:10.2f}%{r['mediana_km']:12.1f}", flush=True)
 
        for key in ["opt2sar", "sar2opt"]:
            enrich = spatial[key]["adiacente"] / max(spatial["caso"]["adiacente"], 1e-9)
            spatial[key]["adiacenti_vs_caso"] = enrich
            print(f"{key}: le tile adiacenti sono il miglior errore {enrich:.0f} volte piu' spesso del caso", flush=True)
        
        same = gid_ev[:, None] == gid_ev[None, :]           
        n_g = same.sum(dim=1)                               
        ok = n_g >= 2                                       
        rank_g = {
            "OPT->SAR": ((S >= diag) & same).sum(dim=1),    
            "SAR->OPT": ((S >= diag.T) & same).sum(dim=0),  
        }
        n_ok = ok.sum().item()
        print(f"\nRETRIEVAL NEL GRUPPO: {n_ok}/{N} serie con almeno un altro candidato nel gruppo | dimensione gruppo mediana {n_g[ok].float().median().item():.0f}, max {n_g[ok].max().item()}", flush=True)
        ingroup = {"n_serie": n_ok}
        for name, rank in rank_g.items():
            r = rank[ok].float()
            n = n_g[ok].float()
            ingroup[name] = {
                "top1": (r <= 1).float().mean().item(),
                "chance_top1": (1.0 / n).mean().item(),
                "top5": (r <= 5).float().mean().item(),
                "chance_top5": (torch.clamp(n, max=5) / n).mean().item(),
                "rank_normalizzato_mediano": ((r - 1) / (n - 1)).median().item(),   # 0 = sempre prima, ~0.5 = a caso
            }
            g = ingroup[name]
            print(f"{name} -> top1: {100 * g['top1']:.2f}% (caso {100 * g['chance_top1']:.2f}%) | top5: {100 * g['top5']:.2f}% (caso {100 * g['chance_top5']:.2f}%) | rank normalizzato mediano: {g['rank_normalizzato_mediano']:.2f} (caso ~0.50)", flush=True)
        results["retrieval_nel_gruppo"] = ingroup
 
        results["spaziale"] = spatial
        print("=" * 50 + "\n", flush=True)
 
    out_path = f"moco_2D_retrieval_{job_name}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)   # nel log, cosi' i numeri restano anche se l'upload fallisce (es. quota)
    try:
        project_work.log_artifact(name=f"moco_2D_retrieval_{job_name}", source=out_path, kind='artifact')
        print("OK -> risultati salvati come artifact", flush=True)
    except Exception as e:
        print(f"EXC -> upload risultati: {e}", flush=True)
 
    return "TERMINATO -> retrieval MoCo 2D"





    