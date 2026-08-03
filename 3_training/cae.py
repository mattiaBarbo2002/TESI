# src/pretrain.py
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

tesi_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cartella_moco = os.path.join(tesi_folder, '2_moco')
sys.path.append(cartella_moco)

# 3. Aggiungila al percorso di sistema
sys.path.append(cartella_moco)

import torchvision.transforms as transforms
from digitalhub_runtime_python import handler

from multimodal_3Dconv_attention import Singlemodal_CAE
from moco.loader import Singlemodal_Loader

# ==========================================
# 1. Helper di lettura e DataLoader Ottimizzato
# ==========================================


# ==========================================
# 2. L'Handler Principale del Job
# ==========================================
@handler()
def pretrain_encoders(
    epochs: int = 200, 
    batch_size: int = 16, 
    lr: float = 1e-4, 
    weight_decay: float = 1e-4,
    patch_size: int = 256,
    n_images1: int = 10,
    n_channels1: int = 2,
    n_images2: int = 20,
    n_channels2: int = 1,
    mamba: bool = False,
    workers: int = 4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n")
    
    # 1. Collegamento ai due progetti separati
    project_work = dh.get_project("BRB_PROGETTO")
    project_data = dh.get_project("IMG_PROGETTO")
    
    # 2. Download dell'artifact dal progetto dati
    print("Scaricamento del dataset Impact_Mesh...")
    dataset_path = project_data.get_artifact("Impact_Mesh").download()
    
    # Adatta questo path se hai una sottocartella "train" come nel tuo main originale
    # es: os.path.join(dataset_path, "dataset", "train")
    traindir = os.path.join(dataset_path, "dataset") 
    
    # Recupera dinamicamente gli ID leggendo i file nella cartella S2 
    # (rimuovendo il suffisso _OPT_t1.zip per ottenere solo "nome_serie")
    s2_dir = os.path.join(traindir, "S2")
    train_data_IDS = list(set([
        f.split('_OPT_')[0] for f in os.listdir(s2_dir) if f.endswith('.zip')
    ]))
    print(f"Trovati {len(train_data_IDS)} ID di serie per l'addestramento.")

    augmentation = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=[-90,90], expand=False, fill=0),
    ]

    # ==========================================
    # PRE-TRAINING S2 (Ottico)
    # ==========================================
    print("\n--- Inizio Pretraining S2 ---")
    modelS2 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)
    optimizerS2 = torch.optim.Adam(modelS2.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss().to(device)
    scalerS2 = torch.cuda.amp.GradScaler()
    best_loss = 9999.0

    datasetS2 = Singlemodal_Loader(train_data_IDS, traindir, transforms.Compose(augmentation),
                                   patch_size, n_images1, n_channels1, 'S2')
    dataloaderS2 = DataLoader(datasetS2, batch_size=batch_size, shuffle=True, 
                              num_workers=workers, pin_memory=True, drop_last=True)
    
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
        
        if epoch_loss < best_loss:
            torch.save(modelS2.state_dict(), 'modelS2_best.pth')
            best_loss = epoch_loss

    # Salvataggio artefatti S2 su Digital Hub
    pd.DataFrame(resultsS2).to_csv('log_pretrainS2.csv', index_label='epoch')
    project_work.log_artifact(name="encoder-s2-weights", path="modelS2_best.pth")
    project_work.log_artifact(name="metrics-s2", path="log_pretrainS2.csv")
    
    del modelS2
    torch.cuda.empty_cache()

    # ==========================================
    # PRE-TRAINING S1 (SAR)
    # ==========================================
    print("\n--- Inizio Pretraining S1 ---")
    modelS1 = Singlemodal_CAE(input_dim=n_channels2, output_dim=10, n_images=n_images2, mamba=mamba).to(device)
    optimizerS1 = torch.optim.Adam(modelS1.parameters(), lr=lr, weight_decay=weight_decay)
    scalerS1 = torch.cuda.amp.GradScaler()
    best_loss = 9999.0

    datasetS1 = Singlemodal_Loader(train_data_IDS, traindir, transforms.Compose(augmentation),
                                   patch_size, n_images2, n_channels2, 'S1')
    dataloaderS1 = DataLoader(datasetS1, batch_size=batch_size, shuffle=True, 
                              num_workers=workers, pin_memory=True, drop_last=True)
    
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
        
        if epoch_loss < best_loss:
            torch.save(modelS1.state_dict(), 'modelS1_best.pth')
            best_loss = epoch_loss

    # Salvataggio artefatti S1 su Digital Hub
    pd.DataFrame(resultsS1).to_csv('log_pretrainS1.csv', index_label='epoch')
    project_work.log_artifact(name="encoder-s1-weights", path="modelS1_best.pth")
    project_work.log_artifact(name="metrics-s1", path="log_pretrainS1.csv")

    return "Pre-training S1 e S2 completato con successo e artefatti registrati."