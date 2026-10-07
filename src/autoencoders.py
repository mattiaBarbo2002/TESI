import os
import zipfile
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import digitalhub as dh
import numpy as np
import matplotlib
import time
import zipfile

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from digitalhub_runtime_python import handler

from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE
from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_2d
from floodlib.loader import Singlemodal_Loader

matplotlib.use("Agg")  


# notebook -> autoencoder_1D

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


# notebook -> autoencoder_2D

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