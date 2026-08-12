import os
import torch
import torch.nn as nn
import pandas as pd
import digitalhub as dh
from tqdm import tqdm
from torch.utils.data import DataLoader
from digitalhub_runtime_python import handler

from multimodal_3Dconv_attention import Singlemodal_CAE
from moco.builder import MoCo2encoders
from moco.loader import MoCo2encodersLoader

# valori uguali al codice originale

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
    print('Using device:', device, flush=True)

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    print("Inizio download dataset", flush=True)
    dataset_path = project_data.get_artifact("Floods_test").download("/data/dataset_mesh")

    s1_dir = os.path.join(dataset_path, "SAR")
    sar_ids = set(f.split('_SAR_')[0] for f in os.listdir(s1_dir) if f.endswith('.zip'))
    s2_dir = os.path.join(dataset_path, "OPT")
    opt_ids = set(f.split('_OPT_')[0] for f in os.listdir(s2_dir) if f.endswith('.zip'))


    # serie completa = 4 img SAR e 4 img ottiche
    train_data_IDS = list(sar_ids & opt_ids)
    print(f"{len(train_data_IDS)} serie complete SAR e OPT", flush=True)


    # caricamento pesi encoders
    print("Caricamento pesi Encoders", flush=True)
    path_s2 = project_work.get_artifact("encoder-s2-weights").download("modelS2_best.pth")
    path_s1 = project_work.get_artifact("encoder-s1-weights").download("modelS1_best.pth")

    modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=10, n_images=n_images2, mamba=mamba).to(device)
    modelS2.load_state_dict(torch.load(path_s2, map_location=device))  

    modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)
    modelS1.load_state_dict(torch.load(path_s1, map_location=device))

    print("Pesi Encoders caricati", flush=True)


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
        print(f"Eccezione MoCo2encodersLoader: {e}", flush=True)  

    try:     
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True,
        )
    except Exception as e:
        print("Eccezione DataLoader: {e}", flush=True)    

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
        print("Eccezzione in MoCo train: {e}", flush=True)

    pd.DataFrame(results).to_csv('log_moco.csv', index_label='epoch')

    try:
        project_work.log_artifact(name="moco-weights", source="moco_best.pth")
        project_work.log_artifact(name="moco-metrics", source="log_moco.csv")

    except Exception as e:
        print("Eccezione in salvataggio metriche MoCo: {e}", flush=True)

    return "MoCo training finito"

