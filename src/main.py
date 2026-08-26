
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
from moco.loader import Singlemodal_Loader

from moco.builder import MoCo2encoders
from moco.loader import MoCo2encodersLoader

# from anomaly import PairLoader

# main.py contiene tutti gli handler, il file DEVE chiamarsi main.py 

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
        modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)
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

        try:
            dataloaderS1 = DataLoader(datasetS1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        except Exception as e:
            print(f"EXC -> Eccezione in DataLoader S1 {e}", flush=True)

        resultsS1 = {'lr': [], 'train_loss': []}
        epochs_no_improve = 0

        try:
            for epoch in range(1, epochs + 1):
                modelS1.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS1)

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
                    torch.save(state_dict, f'modelS1_best_{dataset}_{job_name}.pth')
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                    pd.DataFrame(resultsS1).to_csv(f'log_pretrainS1_{dataset}_{job_name}.csv', index_label='epoch')
                    try:
                        project_work.log_artifact(name=f"encoder-s1-weights_{dataset}_{job_name}", source=f'modelS1_best_{dataset}_{job_name}.pth')
                    except Exception as e:    
                        print(f"EXC -> upload pesi S1: {e}", flush=True)
                    try:    
                        project_work.log_artifact(name=f"metrics-s1_{dataset}_{job_name}", source=f'log_pretrainS1_{dataset}_{job_name}.csv')
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

        print("OK -> Terminato training S1", flush=True)     


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
        modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=10, n_images=n_images2, mamba=mamba).to(device)
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
                    torch.save(state_dict, f'modelS2_best_{dataset}_{job_name}.pth')
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)

                    pd.DataFrame(resultsS2).to_csv(f'log_pretrainS2_{dataset}_{job_name}.csv', index_label='epoch')
                    try:
                        project_work.log_artifact(name=f"encoder-s2-weights_{dataset}_{job_name}", source=f'modelS2_best_{dataset}_{job_name}.pth')
                    except Exception as e:
                        print(f"EXC -> upload pesi S2: {e}", flush=True)
                    try:
                        project_work.log_artifact(name=f"metrics-s2_{dataset}_{job_name}", source=f'log_pretrainS2_{dataset}_{job_name}.csv')
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

        del modelS2
        torch.cuda.empty_cache()
        print("OK -> Terminato training S2", flush=True)

        print("OK -> Terminato training S2\n")    
    
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
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    print(f"Download: {dataset}", flush=True)
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
    with open(f'train_ids_{dataset}_{job_name}.json', 'w') as f:
        json.dump(train_ids, f)
    try:
        project_work.log_artifact(name=f"moco-train-ids_{dataset}_{job_name}", source=f'train_ids_{dataset}_{job_name}.json')
    except Exception as e:
        print(f"EXC -> upload train ids fallito: {e}", flush=True)

    with open(f'test_ids_{dataset}_{job_name}.json', 'w') as f:
        json.dump(held_out_ids, f)
    try:
        project_work.log_artifact(name=f"moco-test-ids_{dataset}_{job_name}", source=f'test_ids_{dataset}_{job_name}.json')
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
            root=dataset_path,
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
            for im_q, im_k in train_bar:
                im_q = im_q.to(non_blocking=True, device=device)
                im_k = im_k.to(non_blocking=True, device=device)

                with torch.cuda.amp.autocast():
                    loss = model(im_q, im_k)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_num += batch_size
                total_loss += loss.item() * batch_size
                train_bar.set_description(f'MoCo Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')

            epoch_loss = total_loss / total_num
            results['lr'].append(optimizer.param_groups[0]['lr'])
            results['train_loss'].append(epoch_loss)

            if epoch_loss < best_loss - min_delta:
                torch.save(model.state_dict(), f'moco_best_{dataset}_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0

                pd.DataFrame(results).to_csv(f'log_moco_{dataset}_{job_name}.csv', index_label='epoch')

                try:
                    project_work.log_artifact(name=f"moco-weights_{dataset}_{job_name}", source=f'moco_best_{dataset}_{job_name}.pth')
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", flush=True)

                try:
                    project_work.log_artifact(name=f"moco-metrics_{dataset}_{job_name}", source=f'log_moco_{dataset}_{job_name}.csv')
                except Exception as e:
                    print(f"EXC -> upload metriche MoCo: {e}", flush=True)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early Stopping MoCo: epoch {epoch}")
                    break

    except Exception as e:
        print(f"EXC -> Eccezione in MoCo train: {e}", flush=True)

    print("OK -> MoCo training finito", flush=True)
    
    return "TERMINATO -> MoCo training finito"


# handler s1, s2 separati

"""
@handler()
def pretrain_encoder_s1(
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
    patience: int = 20,
    job_name: str = "nome_job",
    dataset: str = "Test",
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
    print(f"Download dataset: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")
    print("OK -> Download terminato")

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

    except Exception as e:
        print(f"EXC -> Eccezione recupero liste: {e}", flush=True)    


    # SAR
    print("\n Training S1")
    modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)
    if n_gpus > 1:
        modelS1 = nn.DataParallel(modelS1)
    optimizerS1 = torch.optim.Adam(modelS1.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss().to(device)
    scalerS1 = torch.cuda.amp.GradScaler()
    best_loss = 9999.0

    try:
        datasetS1 = Singlemodal_Loader(
            listIDs=train_data_SAR_IDS,
            root=dataset_path,
            zip_map=sar_zip_map,
            transform=None,
            patch_size=patch_size,
            n_images=n_images1,
            n_channels=n_channels1,
            data_type='SAR'
        )

        for zip_path in set(sar_zip_map.values()):
            if os.path.exists(zip_path):
                os.remove(zip_path)
        print("OK -> ZIP SAR eliminati", flush=True)

    except Exception as e:
        print(f"EXC -> Eccezione in Loader S1:", {e}, flush=True)   

    try:
        dataloaderS1 = DataLoader(datasetS1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)

    except Exception as e:
        print(f"EXC -> Eccezione in DataLoader S1 {e}", flush=True)     

    resultsS1 = {'lr': [], 'train_loss': []}
    epochs_no_improve = 0

    
    print("\n Training S1")
    modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)
    if n_gpus > 1:
        modelS1 = nn.DataParallel(modelS1)
    optimizerS1 = torch.optim.Adam(modelS1.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss().to(device)
    scalerS1 = torch.cuda.amp.GradScaler()
    best_loss = float('inf')
    min_delta = 1e-4

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

    try:
        dataloaderS1 = DataLoader(datasetS1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    except Exception as e:
        print(f"EXC -> Eccezione in DataLoader S1 {e}", flush=True)

    resultsS1 = {'lr': [], 'train_loss': []}
    epochs_no_improve = 0

    try:
        for epoch in range(1, epochs + 1):
            modelS1.train()
            total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS1)

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
                torch.save(state_dict, f'modelS1_best_{dataset}_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0

                pd.DataFrame(resultsS1).to_csv(f'log_pretrainS1_{dataset}_{job_name}.csv', index_label='epoch')
                try:
                    project_work.log_artifact(name=f"encoder-s1-weights_{dataset}_{job_name}", source=f'modelS1_best_{dataset}_{job_name}.pth')
                except Exception as e:
                    print(f"EXC -> upload pesi S1 fallito: {e}", flush=True)

                try:  
                    project_work.log_artifact(name=f"metrics-s1_{dataset}_{job_name}", source=f'log_pretrainS1_{dataset}_{job_name}.csv')
                except Exception as e:
                    print(f"EXC -> upload intermedio S1 fallito: {e}", flush=True)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early Stopping S1: epoch {epoch}")
                    break

    except Exception as e:
        print(f"EXC -> Eccezione in model train S1: {e}", flush=True)

    del modelS1
    torch.cuda.empty_cache()
    print("OK -> Terminato training S1", flush=True)
       
    return "TERMINATO -> training SAR finito"


@handler()
def pretrain_encoder_s2(
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
    patience: int = 20,
    job_name: str = "nome_job",
    dataset: str = "Test",
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
    print(f"Download dataset: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")
    print("OK -> Download terminato")

    try:
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


    print("\n Training S2")
    modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=10, n_images=n_images2, mamba=mamba).to(device)
    if n_gpus > 1:
        modelS2 = nn.DataParallel(modelS2)
    optimizerS2 = torch.optim.Adam(modelS2.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss().to(device)
    scalerS2 = torch.cuda.amp.GradScaler()
    best_loss = float('inf')
    min_delta = 1e-4

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
                torch.save(state_dict, f'modelS2_best_{dataset}_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0

                pd.DataFrame(resultsS2).to_csv(f'log_pretrainS2_{dataset}_{job_name}.csv', index_label='epoch')
                try:
                    project_work.log_artifact(name=f"encoder-s2-weights_{dataset}_{job_name}", source=f'modelS2_best_{dataset}_{job_name}.pth')
                except Exception as e:
                    print(f"EXC -> upload pesi S2 fallito: {e}", flush=True)

                try:
                    project_work.log_artifact(name=f"metrics-s2_{dataset}_{job_name}", source=f'log_pretrainS2_{dataset}_{job_name}.csv')
                except Exception as e:
                    print(f"EXC -> upload metriche S2 fallito: {e}", flush=True)    

            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early Stopping S2: epoch {epoch}")
                    break

    except Exception as e:
        print(f"EXC -> Eccezione in model train S2: {e}", flush=True)

    del modelS2
    torch.cuda.empty_cache()
    print("OK -> Terminato training S2", flush=True)
   
    

    return "TERMINATO -> training OPT finito"

"""

