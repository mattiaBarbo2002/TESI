import os
import zipfile
import torch
import torch.nn as nn 
import numpy as np   
import pandas as pd
import digitalhub as dh
import matplotlib
import random
import json
import time
import zipfile
import math
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from digitalhub_runtime_python import handler

from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE
from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_2d
from floodlib.multimodal_3Dconv_attention import Singlemodal_CAE_attn

from floodlib.loader import MoCo2encodersLoader

from floodlib.builder import MoCo2encoders
from floodlib.builder import MoCo2encoders_2d
from floodlib.builder import MoCo2encoders_Bot
from floodlib.builder import MoCo2encoders_attn
from floodlib.builder import MoCo2encoders_ms
from floodlib.builder import MoCo2encoders_ms_pre

from floodlib.mocoDense import MoCo2encoders_dense
from floodlib.mocoDense import MoCo2encoders_dense3


from floodlib.bottleneck import MocoBottleneck
from floodlib.bottleneck import MocoBottleneckProj


matplotlib.use("Agg")  


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
def train_moco_bot(
    job_name: str = "nome_job",
    dataset: str = "Test",
    weights_encoder_sar: str = "train_sar_1D_v1_Standard_200",
    weights_encoder_opt: str = "train_opt_1D_v1_Standard_200",
    hidden_channels_dim: int = 64,
    simple_proj: bool = True,
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
        project_work.log_artifact(name=f"moco_bot_trainIdsList_{job_name}", source=f'train_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload train_ids: {e}", flush=True)
 
    with open(f'test_ids_{job_name}.json', 'w') as f:
        json.dump(test_ids, f)
    try:
        project_work.log_artifact(name=f"moco_bot_testIdsList_{job_name}", source=f'test_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload test: {e}", flush=True)
 
    # caricamento pesi encoders
    try:
        path_sar = project_work.get_artifact(weights_encoder_sar).download(f"{weights_encoder_sar}.pth")
        path_opt = project_work.get_artifact(weights_encoder_opt).download(f"{weights_encoder_opt}.pth")
 
        modelSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
        modelSAR.load_state_dict(torch.load(path_sar, map_location=device))
 
        modelOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)
        modelOPT.load_state_dict(torch.load(path_opt, map_location=device))
        print(f"OK -> Pesi encoders caricati: {weights_encoder_sar}, {weights_encoder_opt}", "\n", flush=True)
    except Exception as e:
        print(f"EXC -> upload pesi encoders: {e}", "\n", flush=True)  
 
 
    # TRAINING MOCO 2D
    
    print("--- Training MoCo Bot ---")
    model = MoCo2encoders_Bot(
        base_encoder_q=modelOPT.encoder,
        base_encoder_k=modelSAR.encoder,
        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
        symmetric=symmetric, device=device,
        hidden_channels_dim=hidden_channels_dim,
        pool_grid=pool_grid,
        simple_proj=simple_proj,
    ).to(device)

    print(model.conv_q, flush=True)                                                  
    print(f"parametri testa q: {sum(p.numel() for p in model.conv_q.parameters())}", flush=True)   
 
    results = {'lr': [], 'train_loss': []}
    best_loss = float('inf')
    epochs_no_improve = 0
 
    # variabili per caricamento pesi
    start_epoch = 1
    queue_restored = False
 
    # resume = True -> carico pesi vecchi di moco e coda già inzializzata
    if resume:
        try:
            w_path = project_work.get_artifact(f"moco_bot_weights_{job_name}").download("/data")
            state_dict = torch.load(w_path, map_location=device)
            model.load_state_dict(state_dict)
            queue_restored = True
            print("OK -> pesi MoCo caricati", flush=True)
        except Exception as e:
            print(f"EXC -> no pesi MoCo vecchi, inizializzazione casuale: {e}", flush=True)
 
        try:
            m_path = project_work.get_artifact(f"moco_bot_metrics_{job_name}").download("/data")
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
    steps_total = None   
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
            pd.DataFrame(results).to_csv(f'moco_bot_log_{job_name}.csv', index_label='epoch')
 
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
                project_work.log_artifact(name=f"moco_bot_metrics_{job_name}", source=f'moco_bot_log_{job_name}.csv', kind='artifact')
                print(f"OK -> metriche MoCo salvate", flush=True)
            except Exception as e:
                print(f"EXC -> upload metriche MoCo: {e}", flush=True)
 
            # skip first epoch for patience
            skip_patience_epoch = (epoch == start_epoch) and (not queue_restored)
 
            if skip_patience_epoch:
                torch.save(model.state_dict(), f'moco_bot_model_best_{job_name}.pth')
                print(f"skip patience epoch {epoch}", flush=True)
                print(f"loss skipped epoch {epoch}: {best_loss}", flush=True)
 
                try:
                    project_work.log_artifact(name=f"moco_bot_weights_{job_name}", source=f'moco_bot_model_best_{job_name}.pth', kind='artifact')
                    print(f"OK -> pesi salvati\n", flush=True)
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", "\n", flush=True)
 
            elif epoch_loss < best_loss - min_delta:
                torch.save(model.state_dict(), f'moco_bot_model_best_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0
                print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)
 
                try:
                    project_work.log_artifact(name=f"moco_bot_weights_{job_name}", source=f'moco_bot_model_best_{job_name}.pth', kind='artifact')
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
    
    return "TERMINATO -> training moco bot"

@handler()
def prova(
    job_name: str = "",
):
    PROJECT_NAME = "floods"
    project = dh.get_project(PROJECT_NAME)
    print(f"Project: {project.name}")

    p = project.get_artifact("moco_2D_weights_dense_v2").download("/tmp/dense.pth", overwrite=True)
    sd = torch.load(p, map_location="cpu")

    def stato_coda(Q, nome):
        """Q: (dim, K), un vettore normalizzato per colonna."""
        S = Q.T @ Q                                                    # similarità tra tutte le coppie
        fuori = S[~torch.eye(S.shape[0], dtype=torch.bool)]            # esclude la diagonale (sé stessi)
        lam = torch.linalg.svdvals(Q) ** 2
        rango_eff = (lam.sum() ** 2 / (lam ** 2).sum()).item()          # in quante direzioni si distribuiscono
        print(f"{nome:13s} | similarità media {fuori.mean():.3f} ± {fuori.std():.3f} | "
            f"dimensionalità effettiva {rango_eff:.1f} su {Q.shape[0]}")

    stato_coda(sd["queue"], "coda globale")
    stato_coda(sd["queue2"], "coda densa")


@handler()
def diagnosi_dense(
    moco_version: str = "net_dense_v4",                                         
    dataset: str = "Test",
    hidden_channels_dim: int = 128,
    pool_grid: int = 4,
    moco_dim: int = 128,
    moco_k: int = 4096,
    grid_dense: int = 8,                                                       
    T_dense: float = 0.2,                                                       
    n_neg: int = 4,                                                             
    batch_size: int = 16,
    n_batch: int = 20,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    # dimensionalità effettiva di un insieme di vettori (righe): in quante direzioni si distribuiscono
    def dim_effettiva(v):
        lam = torch.linalg.svdvals(v.float()) ** 2
        return (lam.sum() ** 2 / (lam ** 2).sum()).item()

    # similarità media tra vettori diversi (righe normalizzate)
    def similarita_media(v):
        S = v @ v.T
        fuori = S[~torch.eye(S.shape[0], dtype=torch.bool, device=S.device)]
        return fuori.mean().item(), fuori.std().item()

    # ---------- dati: dataset piccolo, estratto tutto ----------
    print(f"Download dataset: Floods_{dataset}_crop_norm", flush=True)
    path = project_data.get_artifact(f"Floods_{dataset}_crop_norm").download("/data/ds")
    for sensor, cache in (("SAR", "/data/cache_SAR"), ("OPT", "/data/cache_OPT")):
        os.makedirs(cache, exist_ok=True)
        for part in sorted(glob(os.path.join(path, sensor, "part*.zip"))):
            with zipfile.ZipFile(part) as z:
                z.extractall(cache)
    ids_sar = {f[:-4] for f in os.listdir("/data/cache_SAR") if f.endswith(".npy")}
    ids_opt = {f[:-4] for f in os.listdir("/data/cache_OPT") if f.endswith(".npy")}
    ids = sorted(ids_sar & ids_opt)[: batch_size * n_batch]
    print(f"OK -> {len(ids)} serie per il test", "\n", flush=True)

    
    enc_opt = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False)
    enc_sar = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False)
    model = MoCo2encoders_dense3(                                               
        base_encoder_q=enc_opt.encoder,
        base_encoder_k=enc_sar.encoder,
        dim=moco_dim, K=moco_k,
        hidden_channels_dim=hidden_channels_dim,
        pool_grid=pool_grid,
        n_neg=n_neg, grid_dense=grid_dense, T_dense=T_dense,                    
    ).to(device)

    try:
        w = project_work.get_artifact(f"moco_2D_weights_{moco_version}").download("/data/w.pth", overwrite=True)
        model.load_state_dict(torch.load(w, map_location=device))
        print(f"OK -> modello caricato: {moco_version}", "\n", flush=True)
    except Exception as e:
        print(f"EXC -> caricamento modello: {e}", flush=True)
        raise

    # ---------- test 1: stato delle code ----------
    print("===== TEST 1: code =====", flush=True)
    for nome, Q in (("coda globale", model.queue), ("coda densa", model.queue2)):
        v = Q.T                                                     # (K, dim): un vettore per riga
        m, s = similarita_media(v)
        print(f"{nome:13s} | similarità media {m:.3f} ± {s:.3f} | dimensionalità effettiva {dim_effettiva(v):.1f} su {v.shape[1]}", flush=True)
    print(flush=True)

    # ---------- dataloader ----------
    loader = DataLoader(
        MoCo2encodersLoader(listIDs=ids, sar_map={}, opt_map={}, transform=None, patch_size=patch_size,
                            n_images1=n_images1, n_channels1=n_channels1,
                            n_images2=n_images2, n_channels2=n_channels2),
        batch_size=batch_size, shuffle=False, drop_last=True,
    )

    # come nel training: testa in train, encoder in eval
    model.train()
    model.encoder_q.eval()
    model.encoder_k.eval()

    params = [p for p in model.conv_q.parameters() if p.requires_grad]     # conv in comune tra i due rami
    piatto = lambda gs: torch.cat([g.flatten() for g in gs])

    coseni, rapporti = [], []
    dim_pos_q, dim_pos_k, dim_medie, sim_pos, sim_stessa_scena = [], [], [], [], []

    print("===== TEST 2: gradienti e vettori densi =====", flush=True)
    for i, (im_q, im_k) in enumerate(loader):
        im_q = im_q.to(device)
        im_k = im_k.to(device)

        # --- conflitto tra i gradienti sulle conv in comune ---
        loss, loss_g, loss_d, _, _ = model.contrastive_loss(im_q, im_k)
        g_g = piatto(torch.autograd.grad(loss_g, params, retain_graph=True))   # gradiente della sola loss globale
        g_d = piatto(torch.autograd.grad(loss_d, params))                      # gradiente della sola loss densa
        cos = nn.functional.cosine_similarity(g_g, g_d, dim=0).item()
        rapporto = (g_d.norm() / g_g.norm()).item()
        coseni.append(cos)
        rapporti.append(rapporto)

        # --- vettori densi: per cella e medie per scena ---
        with torch.no_grad():
            cq = model.pool_dense(model.conv_q(model.encoder_q(im_q)))                                 
            ck = model.pool_dense(model.conv_k(model.encoder_k(im_k)))                                 
            dq = nn.functional.normalize(model.dense_q(cq), dim=1)                                   
            dk = nn.functional.normalize(model.dense_k(ck), dim=1)                                     
            B, D, H, W = dq.shape

            # vettori delle singole celle: con 8x8 sono B*64 vettori, li uso tutti       
            vq = dq.permute(0, 2, 3, 1).reshape(-1, D)
            vk = dk.permute(0, 2, 3, 1).reshape(-1, D)
            dim_pos_q.append(dim_effettiva(vq))                                                        
            dim_pos_k.append(dim_effettiva(vk))                                                       

            # medie per scena: un vettore per serie
            medie = nn.functional.normalize(dk.mean(dim=(2, 3)), dim=1)                               # (B, D)
            dim_medie.append(dim_effettiva(medie))

            # similarità positiva: ottico e sar nella stessa cella
            sim_pos.append((dq * dk).sum(dim=1).mean().item())

            # similarità tra celle diverse della stessa scena: tutte le H*W celle               
            v_scena = dk.flatten(2).permute(0, 2, 1)                                                  # (B, H*W, D)
            S = v_scena @ v_scena.transpose(1, 2)                                                     # (B, H*W, H*W)
            maschera = ~torch.eye(H * W, dtype=torch.bool, device=device)
            sim_stessa_scena.append(S[:, maschera].mean().item())

            # ReLU del proiettore denso: ora è in posizione 2 (dopo conv e batchnorm)            
            h = model.dense_q[2](model.dense_q[1](model.dense_q[0](cq)))
            frac_zero = (h == 0).float().mean().item()
            canali_morti = int((h.amax(dim=(0, 2, 3)) == 0).sum())
            print(f"ReLU proiettore denso: {frac_zero:.1%} valori a zero | canali morti {canali_morti} su {h.shape[1]}")

            # parte comune e parte che varia dell'uscita (dopo la batchnorm finale)
            r = model.dense_q(cq)                                                                      
            v = r.permute(0, 2, 3, 1).reshape(-1, r.shape[1])
            media = v.mean(dim=0)
            variazione = (v - media).norm(dim=1).mean()
            print(f"|parte comune| {media.norm():.3f} | |parte che varia| {variazione:.3f} | "
                  f"|W ultima conv| {model.dense_q[3].weight.norm():.3f}")                           

        print(f"batch {i:2d} | loss_g {loss_g.item():.3f} | loss_d {loss_d.item():.3f} | "
              f"coseno grad {cos:+.3f} | |g_d|/|g_g| {rapporto:.2f} | "
              f"dim pos q {dim_pos_q[-1]:.1f} k {dim_pos_k[-1]:.1f} | sim pos {sim_pos[-1]:.3f}", flush=True)

    print("\n===== RIEPILOGO =====", flush=True)
    print(f"coseno tra i gradienti (globale vs densa):       {np.mean(coseni):+.3f} ± {np.std(coseni):.3f}", flush=True)
    print(f"rapporto norme |g_densa| / |g_globale|:          {np.mean(rapporti):.2f}", flush=True)
    print(f"dimensionalità effettiva per cella (ottico):     {np.mean(dim_pos_q):.1f} su {moco_dim}", flush=True)
    print(f"dimensionalità effettiva per cella (sar):        {np.mean(dim_pos_k):.1f} su {moco_dim}", flush=True)
    print(f"dimensionalità effettiva delle medie per scena:  {np.mean(dim_medie):.1f} (su {batch_size} scene per batch)", flush=True)
    print(f"similarità positiva media (stessa cella):        {np.mean(sim_pos):.3f}", flush=True)
    print(f"similarità tra celle della stessa scena:         {np.mean(sim_stessa_scena):.3f}", flush=True)

    return "TERMINATO -> diagnosi dense"  

@handler()
def train_moco_2D(
    job_name: str = "nome_job",
    dataset: str = "Test",
    weights_encoder_sar: str = "train_sar_1D_v1_Standard_200",
    weights_encoder_opt: str = "train_opt_1D_v1_Standard_200",
    hidden_channels_dim: int = 64,
    simple_proj: bool = True,
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
    init_from: str = "",
    weights_version: str = "",
    metrics_version: str = "",
    
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
 
        modelSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
        modelSAR.load_state_dict(torch.load(path_sar, map_location=device))
 
        modelOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)
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
    ).to(device)
 
    results = {'lr': [], 'train_loss': []}
    best_loss = float('inf')
    epochs_no_improve = 0
 
    # variabili per caricamento pesi
    start_epoch = 1
    queue_restored = False

    if init_from:
        p = project_work.get_artifact(init_from).download(f"/data/{init_from}.pth", overwrite=True)
        state_dict = torch.load(p, map_location=device)
        mancanti, inattesi = model.load_state_dict(state_dict, strict=False)
        print(f"non caricati: {mancanti}", flush=True)
        print(f"inattesi: {inattesi}", flush=True)

        # ammessi: chiavi nuove della classe (coda e centri) e coda vecchia della v5
        nuove_ammesse = ("queue2", "queue2_ptr", "center_q", "center_k")
        vecchie_ammesse = ("queue", "queue_ptr")
        if any(k not in nuove_ammesse for k in mancanti) or any(k not in vecchie_ammesse for k in inattesi):
            raise ValueError(f"i pesi di {init_from} non corrispondono alla classe {type(model).__name__}")
        queue_restored = True
        print(f"OK -> pesi iniziali caricati da {init_from}", "\n", flush=True)
 
    # resume = True -> carico pesi vecchi di moco e coda già inzializzata
    if resume:
        try:
            if weights_version:
                art_w = dh.get_artifact(f"moco_2D_weights_{job_name}", project="floods", entity_id=weights_version)
            else:
                art_w = project_work.get_artifact(f"moco_2D_weights_{job_name}")
            w_path = art_w.download("/data", overwrite=True)
            state_dict = torch.load(w_path, map_location=device)
            model.load_state_dict(state_dict)
            queue_restored = True
            print(f"OK -> pesi MoCo caricati (versione {weights_version or 'ultima'})", flush=True)
        except Exception as e:
            print(f"EXC -> resume: pesi MoCo non caricati: {e}", flush=True)
            raise

        try:
            if metrics_version:
                art_m = dh.get_artifact(f"moco_2D_metrics_{job_name}", project="floods", entity_id=metrics_version)
            else:
                art_m = project_work.get_artifact(f"moco_2D_metrics_{job_name}")
            m_path = art_m.download("/data", overwrite=True)
            prev_df = pd.read_csv(m_path)
            results = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}

            start_epoch = len(prev_df) + 1
            valid_losses = prev_df['train_loss'].iloc[1:]
            best_loss = valid_losses.min() if len(valid_losses) > 0 else float('inf')

            print(f"OK -> metriche MoCo caricate (versione {metrics_version or 'ultima'}): "
                  f"{len(prev_df)} epoche, best_loss={best_loss}", flush=True)
        except Exception as e:
            print(f"EXC -> resume: metriche MoCo non caricate: {e}", flush=True)
            raise
 
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
 
    # prima epoca loss molto bassa perche coda inizializzata casualmente
    # lr parte da 0 e poi si stabilizza al valore fissato
    # con batch size = 16 e coda (moco_k) = 4096, la coda si riempie con 256 batch
    # una epoca contiene 1243 batch
 
    epoch_lenght = None
    steps_total = None   
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
def train_moco_dense(
    job_name: str = "nome_job",
    dataset: str = "Test",
    weights_encoder_sar: str = "train_sar_1D_v1_Standard_200",
    weights_encoder_opt: str = "train_opt_1D_v1_Standard_200",
    hidden_channels_dim: int = 64,
    simple_proj: bool = True,
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
    dense_lambda: float = 0.5,          
    n_pos: int = 64,
    n_neg: int = 4,
    grid_dense: int = 8,
    T_dense: float = 0.2,
    
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
 
        modelSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
        modelSAR.load_state_dict(torch.load(path_sar, map_location=device))
 
        modelOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)
        modelOPT.load_state_dict(torch.load(path_opt, map_location=device))
        print(f"OK -> Pesi encoders caricati: {weights_encoder_sar}, {weights_encoder_opt}", "\n", flush=True)
    except Exception as e:
        print(f"EXC -> upload pesi encoders: {e}", "\n", flush=True)  
 
 
    # TRAINING MOCO 2D
    
    print("--- Training MoCo 2D dense ---")
    model = MoCo2encoders_dense3(
        base_encoder_q=modelOPT.encoder,
        base_encoder_k=modelSAR.encoder,
        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
        hidden_channels_dim=hidden_channels_dim,
        pool_grid=pool_grid,
        dense_lambda=dense_lambda,              # fisso per tutto il training
        n_pos=n_pos, n_neg=n_neg,
        grid_dense=grid_dense, T_dense=T_dense,
    ).to(device)
 
    results = {'lr': [], 'train_loss': [], 'loss_globale': [], 'loss_densa': []}
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
            results = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist(), 'loss_globale': prev_df['loss_globale'].tolist(), 'loss_densa': prev_df['loss_densa'].tolist()}
 
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
    steps_total = None   
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
            total_loss_g, total_loss_d = 0.0, 0.0
 
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
                    loss, loss_g, loss_d = model(im_q, im_k)
 
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
                total_loss_g += loss_g.item() * batch_size
                total_loss_d += loss_d.item() * batch_size
                train_bar.set_description(f'MoCo Epoch: [{epoch}/{target_epoch}], Loss: {loss.item():.4f} '
                                          f'(g {loss_g.item():.3f}, d {loss_d.item():.3f})')
 
            epoch_loss = total_loss / total_num
            results['lr'].append(optimizer.param_groups[0]['lr'])
            results['train_loss'].append(epoch_loss)
            results['loss_globale'].append(total_loss_g / total_num)
            results['loss_densa'].append(total_loss_d / total_num)
 
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
def train_moco_attn(
    job_name: str = "nome_job",
    dataset: str = "Test",
    weights_encoder_sar: str = "autoencoder_attn_SAR_weights_v1",
    weights_encoder_opt: str = "autoencoder_attn_OPT_weights_v1",
    hidden_channels_dim: int = 64,
    attn: bool = False,
    ltae: bool = True,
    spac_net: bool = False,
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
        project_work.log_artifact(name=f"moco_attn_trainIdsList_{job_name}", source=f'train_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload train_ids: {e}", flush=True)
 
    with open(f'test_ids_{job_name}.json', 'w') as f:
        json.dump(test_ids, f)
    try:
        project_work.log_artifact(name=f"moco_attn_testIdsList_{job_name}", source=f'test_ids_{job_name}.json', kind='artifact')
    except Exception as e:
        print(f"EXC -> upload test: {e}", flush=True)
 
    # caricamento pesi encoders
    try:
        path_sar = project_work.get_artifact(weights_encoder_sar).download(f"{weights_encoder_sar}.pth")
        path_opt = project_work.get_artifact(weights_encoder_opt).download(f"{weights_encoder_opt}.pth")
 
        modelSAR = Singlemodal_CAE_attn(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=ltae, spac_net=spac_net).to(device)
        modelSAR.load_state_dict(torch.load(path_sar, map_location=device))
 
        modelOPT = Singlemodal_CAE_attn(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=ltae, spac_net=spac_net).to(device)
        modelOPT.load_state_dict(torch.load(path_opt, map_location=device))
        print(f"OK -> Pesi encoders caricati: {weights_encoder_sar}, {weights_encoder_opt}", "\n", flush=True)
    except Exception as e:
        print(f"EXC -> upload pesi encoders: {e}", "\n", flush=True)  
 
 
    # TRAINING MOCO 2D
    
    print("--- Training MoCo 2D ---")
    model = MoCo2encoders_attn(
        base_encoder_q=modelOPT.encoder,
        base_encoder_k=modelSAR.encoder,
        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
        symmetric=symmetric, 
        hidden_channels_dim=hidden_channels_dim,
        pool_grid=pool_grid,
        simple_proj=False,
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
            w_path = project_work.get_artifact(f"moco_attn_weights_{job_name}").download("/data")
            state_dict = torch.load(w_path, map_location=device)
            model.load_state_dict(state_dict)
            queue_restored = True
            print("OK -> pesi MoCo caricati", flush=True)
        except Exception as e:
            print(f"EXC -> no pesi MoCo vecchi, inizializzazione casuale: {e}", flush=True)
 
        try:
            m_path = project_work.get_artifact(f"moco_attn_metrics_{job_name}").download("/data")
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
    steps_total = None  
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
            pd.DataFrame(results).to_csv(f'moco_attn_log_{job_name}.csv', index_label='epoch')
 
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
                project_work.log_artifact(name=f"moco_attn_metrics_{job_name}", source=f'moco_attn_log_{job_name}.csv', kind='artifact')
                print(f"OK -> metriche MoCo salvate", flush=True)
            except Exception as e:
                print(f"EXC -> upload metriche MoCo: {e}", flush=True)
 
            # skip first epoch for patience
            skip_patience_epoch = (epoch == start_epoch) and (not queue_restored)
 
            if skip_patience_epoch:
                torch.save(model.state_dict(), f'moco_attn_model_best_{job_name}.pth')
                print(f"skip patience epoch {epoch}", flush=True)
                print(f"loss skipped epoch {epoch}: {best_loss}", flush=True)
 
                try:
                    project_work.log_artifact(name=f"moco_attn_weights_{job_name}", source=f'moco_attn_model_best_{job_name}.pth', kind='artifact')
                    print(f"OK -> pesi salvati\n", flush=True)
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", "\n", flush=True)
 
            elif epoch_loss < best_loss - min_delta:
                torch.save(model.state_dict(), f'moco_attn_model_best_{job_name}.pth')
                best_loss = epoch_loss
                epochs_no_improve = 0
                print(f"best loss epoch {epoch} (new): {best_loss}", flush=True)
 
                try:
                    project_work.log_artifact(name=f"moco_attn_weights_{job_name}", source=f'moco_attn_model_best_{job_name}.pth', kind='artifact')
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
    
    return "TERMINATO -> training moco attn"




@handler()
def train_moco_ms(
    job_name: str = "net_v5_ms",
    dataset: str = "Standard",
    weights_encoder_sar: str = "autoencoder_2D_SAR_weights_convLstm_v1",
    weights_encoder_opt: str = "autoencoder_2D_OPT_weights_convLstm_v1",
    hidden_channels_dim: int = 128,
    pool_grid: int = 4,
    init_from: str = "moco_2D_weights_net_v5",      # pesi di partenza (es. v5); "" = da zero
    n_center: int = 2000,                           # serie usate per calcolare i centri
    resume: bool = False,
    lr_min: float = 0.0001,
    cosine: bool = True,
    time_debug: bool = False,

    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.002,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    moco_dim: int = 128,
    moco_k: int = 4096,
    moco_m: float = 0.999,
    moco_t: float = 0.07,
    workers: int = 0,
    patience: int = 20,
    min_delta: float = 1e-4,

    init_version: str = "",
    center_m: float = 1.0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n", flush=True)
    print("handler train_moco_ms: controlli prima del dataset", "\n", flush=True)      

    torch.backends.cudnn.benchmark = True

    if init_from and resume:
        raise ValueError("init_from e resume sono alternativi: init_from per una run nuova, resume per riprendere questa")

    # progetti digital hub
    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    
    try:
        path_sar = project_work.get_artifact(weights_encoder_sar).download(f"{weights_encoder_sar}.pth")
        path_opt = project_work.get_artifact(weights_encoder_opt).download(f"{weights_encoder_opt}.pth")

        modelSAR = Singlemodal_CAE_2d(input_dim=n_channels1, output_dim=16, n_images=n_images1, n_head=8, d_k=8, ltae=False).to(device)
        modelSAR.load_state_dict(torch.load(path_sar, map_location=device))

        modelOPT = Singlemodal_CAE_2d(input_dim=n_channels2, output_dim=16, n_images=n_images2, n_head=8, d_k=8, ltae=False).to(device)
        modelOPT.load_state_dict(torch.load(path_opt, map_location=device))
        print(f"OK -> Pesi encoders caricati: {weights_encoder_sar}, {weights_encoder_opt}", "\n", flush=True)
    except Exception as e:
        print(f"EXC -> caricamento pesi encoders: {e}", "\n", flush=True)
        raise

    
    print("--- Training MoCo MS ---", flush=True)
    model = MoCo2encoders_ms_pre(
        base_encoder_q=modelOPT.encoder,
        base_encoder_k=modelSAR.encoder,
        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t,
        hidden_channels_dim=hidden_channels_dim,
        pool_grid=pool_grid,
        center_m=center_m,
    ).to(device)
    print(f"classe: {type(model).__name__} | center_m {center_m}", flush=True)

    results = {'lr': [], 'train_loss': []}
    best_loss = float('inf')
    epochs_no_improve = 0

    start_epoch = 1
    queue_restored = False

   
    if init_from:
        try:
            if init_version:
                art = dh.get_artifact(init_from, project="floods", entity_id=init_version)
            else:
                art = project_work.get_artifact(init_from)
            p = art.download(f"/data/{init_from}.pth", overwrite=True)
            state_dict = torch.load(p, map_location=device)
            mancanti, inattesi = model.load_state_dict(state_dict, strict=False)
            print(f"non caricati: {mancanti}", flush=True)
            print(f"inattesi: {inattesi}", flush=True)

            nuove_ammesse = ("queue2", "queue2_ptr", "center_q", "center_k")
            vecchie_ammesse = ("queue", "queue_ptr")
            mancanti_extra = [k for k in mancanti if k not in nuove_ammesse]
            inattesi_extra = [k for k in inattesi if k not in vecchie_ammesse]
            if mancanti_extra or inattesi_extra:
                print(f"mancanti non ammesse: {mancanti_extra}", flush=True)
                print(f"inattesi non ammessi: {inattesi_extra}", flush=True)
                raise ValueError(f"i pesi di {init_from} non corrispondono alla classe {type(model).__name__}")
            print(f"OK -> pesi iniziali caricati da {init_from} (versione {init_version or 'ultima'})", "\n", flush=True)
        except Exception as e:
            print(f"EXC -> init_from: {e}", flush=True)
            raise

    if resume:
        try:
            w_path = project_work.get_artifact(f"moco_2D_weights_{job_name}").download("/data", overwrite=True)
            model.load_state_dict(torch.load(w_path, map_location=device))
            queue_restored = True
            print("OK -> pesi MoCo caricati", flush=True)
        except Exception as e:
            print(f"EXC -> resume: pesi MoCo non caricati: {e}", flush=True)
            raise

        try:
            m_path = project_work.get_artifact(f"moco_2D_metrics_{job_name}").download("/data", overwrite=True)
            prev_df = pd.read_csv(m_path)
            results = {'lr': prev_df['lr'].tolist(), 'train_loss': prev_df['train_loss'].tolist()}
            start_epoch = len(prev_df) + 1
            valid_losses = prev_df['train_loss'].iloc[1:]
            best_loss = valid_losses.min() if len(valid_losses) > 0 else float('inf')
            print(f"OK -> metriche MoCo caricate: {len(prev_df)} epoche, best_loss={best_loss}", flush=True)
        except Exception as e:
            print(f"EXC -> resume: metriche MoCo non caricate: {e}", flush=True)
            raise

    # download dataset (lento: solo dopo che modello e pesi sono a posto)
    sar_zip_map = {}
    opt_zip_map = {}

    print(f"Download dataset: {dataset}", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}_crop_norm").download("/data/dataset_floods")
    print("OK -> Download terminato", flush=True)

    os.makedirs("/data/cache_SAR", exist_ok=True)
    for part_path in sorted(glob(os.path.join(dataset_path, "SAR", "part*.zip"))):
        with zipfile.ZipFile(part_path, 'r') as z:
            z.extractall("/data/cache_SAR")
    train_data_SAR_IDS = [f[:-4] for f in os.listdir("/data/cache_SAR") if f.endswith('.npy')]
    print(f"OK -> SAR caricato: {len(train_data_SAR_IDS)} serie", flush=True)

    os.makedirs("/data/cache_OPT", exist_ok=True)
    for part_path in sorted(glob(os.path.join(dataset_path, "OPT", "part*.zip"))):
        with zipfile.ZipFile(part_path, 'r') as z:
            z.extractall("/data/cache_OPT")
    train_data_OPT_IDS = [f[:-4] for f in os.listdir("/data/cache_OPT") if f.endswith('.npy')]
    print(f"OK -> OPT caricato: {len(train_data_OPT_IDS)} serie", flush=True)

    # ordinamento
    train_data_IDS = sorted(set(train_data_SAR_IDS) & set(train_data_OPT_IDS))
    print(f"{len(train_data_IDS)} serie complete SAR e OPT", flush=True)

    # split 80/20 (stesso seed della v5: stesse liste)
    random.seed(42)
    shuffled_ids = train_data_IDS.copy()
    random.shuffle(shuffled_ids)

    split_idx = int(len(shuffled_ids) * 0.8)
    train_ids = shuffled_ids[:split_idx]
    test_ids = shuffled_ids[split_idx:]

    print(f"serie train 80%: {len(train_ids)}", flush=True)
    print(f"serie test 20%: {len(test_ids)}", "\n", flush=True)

    # salvataggio liste train e test
    with open(f'train_ids_{job_name}.json', 'w') as f:
        json.dump(train_ids, f)
    project_work.log_artifact(name=f"moco_2D_trainIdsList_{job_name}", source=f'train_ids_{job_name}.json', kind='artifact')

    with open(f'test_ids_{job_name}.json', 'w') as f:
        json.dump(test_ids, f)
    project_work.log_artifact(name=f"moco_2D_testIdsList_{job_name}", source=f'test_ids_{job_name}.json', kind='artifact')

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    base_lr = lr
    batch_step = 0
    target_epoch = start_epoch + epochs - 1

    train_dataset = MoCo2encodersLoader(
        listIDs=train_ids,
        sar_map=sar_zip_map,
        opt_map=opt_zip_map,
        transform=None,
        patch_size=patch_size,
        n_images1=n_images1, n_channels1=n_channels1,
        n_images2=n_images2, n_channels2=n_channels2,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
    )

    # warmup del lr nella prima epoca solo se la coda non è già piena (run nuova)
    epoch_lenght = len(train_loader) if not queue_restored else None
    steps_total = (target_epoch - start_epoch + 1) * len(train_loader)

    # centri delle feature normali, con i pesi di partenza (solo per una run nuova)
    if not resume:
        model.eval()
        somma_q, somma_k, n = 0, 0, 0
        with torch.no_grad(), torch.cuda.amp.autocast():
            for im_q, im_k in train_loader:
                somma_q = somma_q + model.features_q(im_q.to(device)).float().sum(0)
                somma_k = somma_k + model.features_k(im_k.to(device)).float().sum(0)
                n += im_q.shape[0]
                if n >= n_center:
                    break
        model.set_centers(somma_q / n, somma_k / n)
        model.reset_queue()
        print(f"OK -> centri calcolati su {n} serie | |c_q| {model.center_q.norm():.3f} | "
              f"|c_k| {model.center_k.norm():.3f} | coda svuotata", "\n", flush=True)

    # TRAINING
    try:
        for epoch in range(start_epoch, target_epoch + 1):
            model.train()
            model.encoder_q.eval()
            model.encoder_k.eval()

            total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)

            if time_debug:
                t_prev = time.time()

            for im_q, im_k in train_bar:

                # prima epoca (run nuova): lr sale da 0 a lr; poi coseno da lr a lr_min, oppure lr costante
                if epoch_lenght is not None and batch_step < epoch_lenght:
                    batch_lr = base_lr * (batch_step + 1) / epoch_lenght
                    for pg in optimizer.param_groups:
                        pg['lr'] = batch_lr
                elif cosine:
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

                with torch.cuda.amp.autocast():
                    loss = model(im_q, im_k)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if time_debug:
                    torch.cuda.synchronize()
                    print(f"data: {t_data - t_prev:.3f}s | step: {time.time() - t_data:.3f}s", flush=True)
                    t_prev = time.time()

                total_num += batch_size
                total_loss += loss.item() * batch_size
                train_bar.set_description(f'MoCo MS Epoch: [{epoch}/{target_epoch}], Loss: {loss.item():.4f}')

            epoch_loss = total_loss / total_num
            results['lr'].append(optimizer.param_groups[0]['lr'])
            results['train_loss'].append(epoch_loss)
            pd.DataFrame(results).to_csv(f'moco_2D_log_{job_name}.csv', index_label='epoch')

            # refresh token e progetto
            try:
                dh.refresh_token()
                project_work = dh.get_project("floods")
            except Exception as e:
                print(f"EXC -> refresh token / get project: {e}", flush=True)

            try:
                project_work.log_artifact(name=f"moco_2D_metrics_{job_name}", source=f'moco_2D_log_{job_name}.csv', kind='artifact')
                print("OK -> metriche MoCo salvate", flush=True)
            except Exception as e:
                print(f"EXC -> upload metriche MoCo: {e}", flush=True)

            # prima epoca di una run nuova: coda in riempimento, esclusa dalla patience
            skip_patience_epoch = (epoch == start_epoch) and (not queue_restored)

            if skip_patience_epoch or epoch_loss < best_loss - min_delta:
                torch.save(model.state_dict(), f'moco_2D_model_best_{job_name}.pth')
                if skip_patience_epoch:
                    print(f"skip patience epoch {epoch}: loss {epoch_loss:.4f}", flush=True)
                else:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    print(f"best loss epoch {epoch} (new): {best_loss:.4f}", flush=True)
                try:
                    project_work.log_artifact(name=f"moco_2D_weights_{job_name}", source=f'moco_2D_model_best_{job_name}.pth', kind='artifact')
                    print("OK -> pesi salvati", "\n", flush=True)
                except Exception as e:
                    print(f"EXC -> upload pesi MoCo: {e}", "\n", flush=True)
            else:
                epochs_no_improve += 1
                print(f"loss epoch {epoch}: {epoch_loss:.4f} | best: {best_loss:.4f} | senza miglioramenti: {epochs_no_improve}", "\n", flush=True)
                if epochs_no_improve >= patience:
                    print(f"Early Stopping MoCo: epoch {epoch} -> best loss: {best_loss:.4f}", "\n", flush=True)
                    break

    except Exception as e:
        print(f"EXC -> training MoCo: {e}", "\n", flush=True)
        raise

    print(f"OK -> Terminato training MoCo MS, LOSS (InfoNCE): {best_loss:.4f}", "\n", flush=True)
    return "TERMINATO -> training moco ms"


"""
@handler()
def moco_eval(
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
 
    return "TERMINATO -> MoCo Eval"



"""
