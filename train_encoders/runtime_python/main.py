
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

import torchvision.transforms as transforms
from digitalhub_runtime_python import handler

from multimodal_3Dconv_attention import Singlemodal_CAE
from moco.loader import Singlemodal_Loader


# HANDLER
@handler()
def pretrain_encoders(
    epochs: int = 200, 
    batch_size: int = 16, 
    lr: float = 1e-4, 
    weight_decay: float = 1e-4,
    patch_size: int = 256,
    n_images1: int = 4,
    n_channels1: int = 2,
    n_images2: int = 4,
    n_channels2: int = 1,
    mamba: bool = False,
    workers: int = 0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n")
    
    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")
    
    
    print("Inizio download dataset")
    
    # percorso "/data/dataset_mesh" per usare il volume
    dataset_path = project_data.get_artifact("Impact_Mesh").download("/data/dataset_mesh")
    
    # cartella train
    traindir = os.path.join(dataset_path, "dataset") 

    # recupero id serie SAR
    s1_dir = os.path.join(traindir, "S1")
    train_data_SAR_IDS = list(set([
        f.split('_SAR_')[0] for f in os.listdir(s1_dir) if f.endswith('.zip')
    ]))
    print(f"{len(train_data_SAR_IDS)} serie trovate")
    
    # recupero id serie OPT
    s2_dir = os.path.join(traindir, "S2")
    train_data_OPT_IDS = list(set([
        f.split('_OPT_')[0] for f in os.listdir(s2_dir) if f.endswith('.zip')
    ]))
    print(f"{len(train_data_OPT_IDS)} serie trovate")


    # OTTICO
    print("\n Pretraining S2 ---")
    modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=10, n_images=n_images2, mamba=mamba).to(device)
    optimizerS2 = torch.optim.Adam(modelS2.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss().to(device)
    scalerS2 = torch.cuda.amp.GradScaler()
    best_loss = 9999.0

    datasetS2 = Singlemodal_Loader(
        listIDs=train_data_OPT_IDS,
        root=traindir,
        transform=None,          
        patch_size=patch_size,
        n_images=n_images2,
        n_channels=n_channels2,
        data_type='OPT'
    )

    dataloaderS2 = DataLoader(datasetS2, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    
    resultsS2 = {'lr': [], 'train_loss': []}

    for epoch in range(1, epochs + 1):
        modelS2.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS2)
        for im in train_bar:
            im = im.to(non_blocking=True, device=device)
            output = modelS2(im)
            loss = loss_fn(output, im)
            
            optimizerS2.zero_grad()
            scalerS2.scale(loss).backward()
            scalerS2.step(optimizerS2)
            scalerS2.update()
            
            total_num += dataloaderS2.batch_size
            total_loss += loss.item() * dataloaderS2.batch_size
            train_bar.set_description(f'S2 Epoch: [{epoch}/{epochs}], Loss: {loss.item():.4f}')
            
        epoch_loss = total_loss / total_num
        resultsS2['lr'].append(optimizerS2.param_groups[0]['lr'])
        resultsS2['train_loss'].append(epoch_loss)
        
        # salvataggio temporaneo su container
        if epoch_loss < best_loss:
            torch.save(modelS2.state_dict(), 'modelS2_best.pth')
            best_loss = epoch_loss

    # salvataggio su digitalhub
    pd.DataFrame(resultsS2).to_csv('log_pretrainS2.csv', index_label='epoch')
    project_work.log_artifact(name="encoder-s2-weights", path="modelS2_best.pth")
    project_work.log_artifact(name="metrics-s2", path="log_pretrainS2.csv")
    
    del modelS2
    torch.cuda.empty_cache()

    # SAR
    print("\n Pretraining S1")
    modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)
    optimizerS1 = torch.optim.Adam(modelS1.parameters(), lr=lr, weight_decay=weight_decay)
    scalerS1 = torch.cuda.amp.GradScaler()
    best_loss = 9999.0

    datasetS1 = Singlemodal_Loader(
        listIDs=train_data_SAR_IDS,
        root=traindir,
        transform=None,
        patch_size=patch_size,
        n_images=n_images1,
        n_channels=n_channels1,
        data_type='SAR'
    )

    dataloaderS1 = DataLoader(datasetS1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    
    resultsS1 = {'lr': [], 'train_loss': []}

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
            torch.save(modelS1.state_dict(), 'modelS1_best.pth')
            best_loss = epoch_loss

    # salvataggio su digitalhub
    pd.DataFrame(resultsS1).to_csv('log_pretrainS1.csv', index_label='epoch')
    project_work.log_artifact(name="encoder-s1-weights", path="modelS1_best.pth")
    project_work.log_artifact(name="metrics-s1", path="log_pretrainS1.csv")

    return "Pre-training SAR e OPT finito"