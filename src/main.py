
import os
import zipfile
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import rasterio
import digitalhub as dh
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import time

import torchvision.transforms as transforms
from digitalhub_runtime_python import handler

from multimodal_3Dconv_attention import Singlemodal_CAE
from moco.loader import Singlemodal_Loader

from moco.builder import MoCo2encoders
from moco.loader import MoCo2encodersLoader


# MAIN.PY contiene tutti gli handler, il file DEVE chiamarsi main.py 

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
    workers: int = 0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
    
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    # download dataset
    print("Download dataset", flush=True)
    dataset_path = project_data.get_artifact("Floods_test").download("/data/dataset_floods")
    print("OK -> Download terminato")

    # recupero id serie SAR
    s1_dir = os.path.join(dataset_path, "SAR")
    train_data_SAR_IDS = list(set([
        f.split('_SAR_')[0] for f in os.listdir(s1_dir) if f.endswith('.zip')
    ]))
    print(f"{len(train_data_SAR_IDS)} serie SAR trovate")
    # print(train_data_SAR_IDS)
    
    # recupero id serie OPT
    s2_dir = os.path.join(dataset_path, "OPT")
    train_data_OPT_IDS = list(set([
        f.split('_OPT_')[0] for f in os.listdir(s2_dir) if f.endswith('.zip')
    ]))
    print(f"{len(train_data_OPT_IDS)} serie OPT trovate")
    # print(train_data_OPT_IDS)


    # SAR
    print("\n Training S1")
    modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)
    optimizerS1 = torch.optim.Adam(modelS1.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss().to(device)
    scalerS1 = torch.cuda.amp.GradScaler()
    best_loss = 9999.0

    try:
        datasetS1 = Singlemodal_Loader(
            listIDs=train_data_SAR_IDS,
            root=dataset_path,
            transform=None,
            patch_size=patch_size,
            n_images=n_images1,
            n_channels=n_channels1,
            data_type='SAR'
        )

    except Exception as e:
        print(f"EXC -> Eccezione in Loader S1:", {e}, flush=True)   

    try:
        dataloaderS1 = DataLoader(datasetS1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)

    except Exception as e:
        print(f"EXC -> Eccezione in DataLoader S1 {e}", flush=True)     
    
    resultsS1 = {'lr': [], 'train_loss': []}

    try:
        for epoch in range(1, epochs + 1):
            modelS1.train()
            total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS1)

            for im in train_bar:
                im = im.to(non_blocking=True, device=device)
                output = modelS1(im)
                loss = loss_fn(output, im)
                
                optimizerS1.zero_grad()
                scalerS1.scale(loss).backward()
                scalerS1.step(optimizerS1)
                scalerS1.update()
                
                total_num += dataloaderS1.batch_size
                total_loss += loss.item() * dataloaderS1.batch_size
                train_bar.set_description(f'S1 Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')
                
            epoch_loss = total_loss / total_num
            resultsS1['lr'].append(optimizerS1.param_groups[0]['lr'])
            resultsS1['train_loss'].append(epoch_loss)
            
            # salvataggio temporaneo nel container
            if epoch_loss < best_loss:
                torch.save(modelS1.state_dict(), 'modelS1_best_time.pth')
                best_loss = epoch_loss

    except Exception as e:
        print(f"EXC -> Eccezione in model train S1: {e}", flush=True)       


    # salvataggio su digitalhub
    print("OK -> Terminato training S1")   

    del modelS1
    torch.cuda.empty_cache() 


    # OTTICO
    print("\n Training S2")
    modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=10, n_images=n_images2, mamba=mamba).to(device)
    optimizerS2 = torch.optim.Adam(modelS2.parameters(), lr=lr, weight_decay=weight_decay)
    scalerS2 = torch.cuda.amp.GradScaler()
    best_loss = 9999.0

    try:
        datasetS2 = Singlemodal_Loader(
            listIDs=train_data_OPT_IDS,
            root=dataset_path,
            transform=None,          
            patch_size=patch_size,
            n_images=n_images2,
            n_channels=n_channels2,
            data_type='OPT'
        )

    except Exception as e:
        print(f"EXC -> Eccezione in Loader S2:", {e}, flush=True)
        

    try:
        dataloaderS2 = DataLoader(datasetS2, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)

    except Exception as e:
        print(f"EXC -> Eccezione in DataLoader S2 {e}", flush=True)
          
    
    resultsS2 = {'lr': [], 'train_loss': []}


    try:
        for epoch in range(1, epochs + 1):
            modelS2.train()
            total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS2)
            print(train_bar)

            #time
            t_prev = time.time()

            for im in train_bar:

                #time    
                torch.cuda.synchronize()
                t_data = time.time()

                im = im.to(non_blocking=True, device=device)

                #time
                torch.cuda.synchronize()
                t_transfer = time.time()   

                output = modelS2(im)
                loss = loss_fn(output, im)

                #time
                torch.cuda.synchronize()
                t_forward = time.time()
                
                optimizerS2.zero_grad()
                scalerS2.scale(loss).backward()
                scalerS2.step(optimizerS2)
                scalerS2.update()

                #time
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
            
            # salvataggio temporaneo su container
            if epoch_loss < best_loss:
                torch.save(modelS2.state_dict(), 'modelS2_best_time.pth')
                best_loss = epoch_loss

    except Exception as e:
        print(f"EXC -> Eccezione in model train S2: {e}", flush=True)

    print("OK -> Terminato training S2\n")    
    

    # salvataggio su digitalhub
    print("Salvataggio metriche")

    try:
        pd.DataFrame(resultsS1).to_csv('log_pretrainS1_time.csv', index_label='epoch')
        project_work.log_artifact(name="encoder-s1-weights_time", source="modelS1_best_time.pth")
        project_work.log_artifact(name="metrics-s1_time", source="log_pretrainS1_time.csv")

    except Exception as e:
        print(f"EXC -> Eccezione in salvataggio metriche S1: {e}", flush=True)

    print("OK -> Metriche S1 salvate")      


    try:
        pd.DataFrame(resultsS2).to_csv('log_pretrainS2_time.csv', index_label='epoch')
        project_work.log_artifact(name="encoder-s2-weights_time", source="modelS2_best_time.pth")
        project_work.log_artifact(name="metrics-s2_time", source="log_pretrainS2_time.csv")
       
    except Exception as e:
        print(f"EXC -> Eccezione in salvataggio metriche S2: {e}", flush=True)

    print("OK -> Metriche S2 salvate")

    return "TERMINATO -> training SAR e OPT finito"


# notebook -> moco
# valori uguali al codice originale tranne coda moco_k

@handler()
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
    moco_k: int = 1024,                             # da modificare
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    symmetric: bool = False,
    mamba: bool = False,
    workers: int = 0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
    
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    print("Download dataset", flush=True)

    dataset_path = project_data.get_artifact("Floods_test").download("/data/dataset_floods")

    s1_dir = os.path.join(dataset_path, "SAR")
    sar_ids = set(f.split('_SAR_')[0] for f in os.listdir(s1_dir) if f.endswith('.zip'))

    s2_dir = os.path.join(dataset_path, "OPT")
    opt_ids = set(f.split('_OPT_')[0] for f in os.listdir(s2_dir) if f.endswith('.zip'))


    # serie completa = 4 img SAR e 4 img ottiche
    train_data_IDS = list(sar_ids & opt_ids)
    print(f"{len(train_data_IDS)} serie complete SAR e OPT", flush=True)


    # caricamento pesi encoders
    print("Caricamento pesi Encoders", flush=True)
    path_s1 = project_work.get_artifact("encoder-s1-weights").download("modelS1_best.pth")
    path_s2 = project_work.get_artifact("encoder-s2-weights").download("modelS2_best.pth")

    modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)
    modelS1.load_state_dict(torch.load(path_s1, map_location=device))    

    modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=10, n_images=n_images2, mamba=mamba).to(device)
    modelS2.load_state_dict(torch.load(path_s2, map_location=device))  

    print("OK -> Pesi Encoders caricati", flush=True)


    # MoCo con SAR = key e OTTICO = query
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
            listIDs=train_data_IDS,
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

    try:
        for epoch in range(1, epochs + 1):
            model.train()

            # congelamento pesi encoder
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

            if epoch_loss < best_loss:
                torch.save(model.state_dict(), 'moco_best.pth')
                best_loss = epoch_loss

    except Exception as e:
        print(f"EXC -> Eccezione in MoCo train: {e}", flush=True)

    pd.DataFrame(results).to_csv('log_moco.csv', index_label='epoch')

    try:
        project_work.log_artifact(name="moco-weights", source="moco_best.pth")
        project_work.log_artifact(name="moco-metrics", source="log_moco.csv")

    except Exception as e:
        print(f"EXC -> Eccezione in salvataggio metriche MoCo: {e}", flush=True)

    print("OK -> MoCo training finito")    

    return "TERMINATO -> MoCo training finito"