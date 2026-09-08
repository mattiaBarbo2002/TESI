
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
from multimodal_3Dconv_attention import MaskedAutoEncoder

from moco.loader import Singlemodal_Loader
from moco.loader import MoCo2encodersLoader
from moco.loader import PairsLoader

from moco.builder import MoCo2encoders

# from anomaly import PairLoader

# main.py contiene tutti gli handler, il file DEVE chiamarsi main.py 

def log_artifact_safe(project, name, source, retries=3, delay=5):
   
    for attempt in range(1, retries + 1):
        try:
            art = project.log_artifact(name=name, source=source)
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
def pretrain_encoders(
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

        resultsS1 = {'lr': [], 'train_loss': []}
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
                        project_work = dh.get_project("floods")
                        print("OK -> token aggiornato", flush=True)

                    except Exception as e:
                        print(f"EXC -> aggiornamento token: {e}", flush=True)

                    log_artifact_safe(project_work, f"encoder-s1-weights_{job_name}_{dataset}_{epochs}", f'modelS1_best_{job_name}_{dataset}_{epochs}.pth')

                    try:    
                        project_work.log_artifact(name=f"metrics-s1_{job_name}_{dataset}_{epochs}", source=f'log_pretrainS1_{job_name}_{dataset}_{epochs}.csv')
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

        resultsS2 = {'lr': [], 'train_loss': []}
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
                        project_work = dh.get_project("floods")
                    except Exception as e:
                        print(f"EXC -> aggiornamento token: {e}", flush=True)
    
                    log_artifact_safe(project_work, f"encoder-s2-weights_{job_name}_{dataset}_{epochs}", f'modelS2_best_{job_name}_{dataset}_{epochs}.pth')

                    try:
                        project_work.log_artifact(name=f"metrics-s2_{job_name}_{dataset}_{epochs}", source=f'log_pretrainS2_{job_name}_{dataset}_{epochs}.csv')
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
    weights_encoder_sar: str = "",
    weights_encoder_opt: str = "",
    patience: int = 20,
    min_delta: float = 1e-4,
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
        project_work.log_artifact(name=f"moco-train-ids_{job_name}_{dataset}_{epochs}", source=f'train_ids_{job_name}_{dataset}_{epochs}.json')
    except Exception as e:
        print(f"EXC -> upload train ids fallito: {e}", flush=True)

    with open(f'test_ids_{job_name}_{dataset}_{epochs}.json', 'w') as f:
        json.dump(held_out_ids, f)
    try:
        project_work.log_artifact(name=f"moco-test-ids_{job_name}_{dataset}_{epochs}", source=f'test_ids_{job_name}_{dataset}_{epochs}.json')
    except Exception as e:
        print(f"EXC -> upload held-out ids fallito: {e}", flush=True)

    # caricamento pesi encoders
    print("Caricamento pesi Encoders", flush=True)
    path_s1 = project_work.get_artifact(f"encoder-s1-weights_{weights_encoder_sar}").download(f"modelS1_best_{weights_encoder_sar}.pth")
    path_s2 = project_work.get_artifact(f"encoder-s2-weights_{weights_encoder_opt}").download(f"modelS2_best_{weights_encoder_opt}.pth")

    modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)
    modelS1.load_state_dict(torch.load(path_s1, map_location=device))

    modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=10, n_images=n_images2, mamba=mamba).to(device)
    modelS2.load_state_dict(torch.load(path_s2, map_location=device))
    print("OK -> Pesi Encoders caricati", flush=True)

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

            if epoch_loss < best_loss - min_delta:
                torch.save(model.state_dict(), f'moco_best_{job_name}_{dataset}_{epochs}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0

                pd.DataFrame(results).to_csv(f'log_moco_{job_name}_{dataset}_{epochs}.csv', index_label='epoch')

                try:
                    project_work.log_artifact(name=f"moco-weights_{job_name}_{dataset}_{epochs}", source=f'moco_best_{job_name}_{dataset}_{epochs}.pth')
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", flush=True)

                try:
                    project_work.log_artifact(name=f"moco-metrics_{job_name}_{dataset}_{epochs}", source=f'log_moco_{job_name}_{dataset}_{epochs}.csv')
                except Exception as e:
                    print(f"EXC -> upload metriche MoCo: {e}", flush=True)
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
                    project_work.log_artifact(name=f"encoder-mae-weights_{job_name}_{direction}_{dataset}_{epochs}", source=f'model_mae_best_{job_name}_{direction}_{dataset}_{epochs}.pth')
                except Exception as e:    
                    print(f"EXC -> upload pesi mae: {e}", flush=True)
                try:    
                    project_work.log_artifact(name=f"metrics-mae_{job_name}_{direction}_{dataset}_{epochs}", source=f'log_train_mae_{job_name}_{direction}_{dataset}_{epochs}.csv')
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
def test_encoders(
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
    n_samples: int = 10
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

        if test_sar:
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

            random_sar_ids = random.sample(train_data_SAR_IDS, min(n_samples, len(train_data_SAR_IDS))) if train_data_SAR_IDS else []
            print(f"{len(train_data_SAR_IDS)} serie SAR trovate")

        if test_opt:
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

            random_opt_ids = random.sample(train_data_OPT_IDS, min(n_samples, len(train_data_OPT_IDS))) if train_data_OPT_IDS else []
            print(f"{len(train_data_OPT_IDS)} serie OPT trovate")

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
            modelS1.eval() # freeze pesi

            datasetS1 = Singlemodal_Loader(
                listIDs=random_sar_ids, root=dataset_path, zip_map=sar_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images1,
                n_channels=n_channels1, data_type='SAR'
            )
            loaderS1 = torch.utils.data.DataLoader(datasetS1, batch_size=1, shuffle=False)

            print("SAR:", flush=True)
            with torch.no_grad():
                for i, im in enumerate(loaderS1):
                    im = im.to(device)
                    
                    try:
                        latent_vector = modelS1.encoder(im)
                    except AttributeError:
                        print(f"EXC -> nome encoder s1: {e}", flush=True)
                        break
                    
                    vector = latent_vector.cpu().numpy().flatten()
                    print(f"ID: {random_sar_ids[i]} | shape: {list(latent_vector.shape)}")
                    print(f"VEC: {np.round(vector, 4)}\n", flush=True)

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
            modelS2.eval() # freeze pesi

            datasetS2 = Singlemodal_Loader(
                listIDs=random_opt_ids, root=dataset_path, zip_map=opt_zip_map,
                transform=None, patch_size=patch_size, n_images=n_images2,
                n_channels=n_channels2, data_type='OPT'
            )
            loaderS2 = torch.utils.data.DataLoader(datasetS2, batch_size=1, shuffle=False)

            with torch.no_grad():
                for i, im in enumerate(loaderS2):
                    im = im.to(device)
                    
                    try:
                        latent_vector = modelS2.encoder(im)
                    except AttributeError:
                        print(f"EXC -> nome encoder s1: {e}", flush=True)
                        break
                    
                    vector = latent_vector.cpu().numpy().flatten()
                    print(f"ID: {random_opt_ids[i]} | shape: {list(latent_vector.shape)}")
                    print(f"VEC: {np.round(vector, 4)}\n", flush=True)

            del modelS2
            torch.cuda.empty_cache()

    return "TERMINATO"