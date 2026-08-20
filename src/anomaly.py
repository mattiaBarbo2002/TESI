import os
import zipfile
import tarfile
from glob import glob
import torch
import numpy as np
import rasterio
import digitalhub as dh
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from digitalhub_runtime_python import handler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from multimodal_3Dconv_attention import Singlemodal_CAE
from moco.builder import MoCo2encoders


def read_tiff_from_archive(archive_path, expected_tif_name):
    abs_path = os.path.abspath(archive_path)
    ext = os.path.splitext(abs_path)[1].lower()
    vsi_prefix = "/vsitar/" if ext == ".tar" else "/vsizip/"
    vsi_path = f"{vsi_prefix}{abs_path}/{expected_tif_name}"
    with rasterio.open(vsi_path) as src:
        img_data = src.read()
    return np.nan_to_num(img_data, nan=0.0).astype(np.float32)


def normalize_image(img, data_type):
    if data_type == 'OPT':
        img = img / 10000.0
        img = np.clip(img, 0.0, 1.5)
    else:
        img = np.clip(img, -25.0, 0.0)
        img = (img + 25.0) / 25.0
    return img


class PairLoader(Dataset):

    # coppie: im_q = ottico, im_k = sar
    def __init__(self, listIDs, sar_map, opt_map, patch_size=256,
                 n_images1=4, n_channels1=2, n_images2=4, n_channels2=10):
        self.listIDs = listIDs
        self.sar_map = sar_map
        self.opt_map = opt_map
        self.patch_size = patch_size
        self.n_images1 = n_images1
        self.n_channels1 = n_channels1
        self.n_images2 = n_images2
        self.n_channels2 = n_channels2

    def _load_series(self, ID, archive_map, suffix, n_images, n_channels, data_type):
        im = np.zeros((n_channels, n_images, self.patch_size, self.patch_size), dtype=np.float32)
        serie_temporale = []
        archive_path = archive_map[ID]

        for t in range(n_images):
            time_step = t + 1
            base_name = f"{ID}_{suffix}_t{time_step}"
            img_data = read_tiff_from_archive(archive_path, f"{base_name}.tif")
            img_data = normalize_image(img_data, data_type)
            serie_temporale.append(img_data)

        c_tot, h_tot, w_tot = serie_temporale[0].shape
        start_y = max(0, (h_tot - self.patch_size) // 2)
        start_x = max(0, (w_tot - self.patch_size) // 2)
        pad_y = max(0, self.patch_size - h_tot)
        pad_x = max(0, self.patch_size - w_tot)

        for t in range(n_images):
            img = serie_temporale[t]
            c_to_take = min(n_channels, img.shape[0])
            img_cropped = img[:c_to_take, start_y:start_y+self.patch_size, start_x:start_x+self.patch_size]
            if pad_y > 0 or pad_x > 0:
                img_cropped = np.pad(img_cropped, ((0, 0), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0.0)
            im[:c_to_take, t, :, :] = img_cropped

        return im

    def __getitem__(self, index):
        ID = self.listIDs[index]
        im_opt = self._load_series(ID, self.opt_map, 'OPT', self.n_images2, self.n_channels2, 'OPT')
        im_sar = self._load_series(ID, self.sar_map, 'SAR', self.n_images1, self.n_channels1, 'SAR')
        return torch.from_numpy(im_opt), torch.from_numpy(im_sar)   # im_q, im_k

    def __len__(self):
        return len(self.listIDs)


@handler()
def detect_anomalies(
    n_normal_treshold: int = 500,                   # quante serie normali usare per calcolare la soglia
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    moco_dim: int = 128, moco_k: int = 1024, moco_m: float = 0.999, moco_t: float = 0.07,
    symmetric: bool = False, mamba: bool = False,
    batch_size: int = 8, workers: int = 0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print('Using device:', device, flush=True)

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    # import dataset normale
    print("Download dataset normale", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")

    def build_zip_map(base_dir, prefix):
        m = {}
        for zf in sorted(glob(os.path.join(base_dir, f"{prefix}_*.zip"))):
            with zipfile.ZipFile(zf, 'r') as z:
                for n in z.namelist():
                    if n.lower().endswith('.tif'):
                        ID = os.path.basename(n).split(f'_{prefix}_')[0]
                        m[ID] = zf
        return m

    sar_map = build_zip_map(os.path.join(dataset_path, "SAR"), "SAR")
    opt_map = build_zip_map(os.path.join(dataset_path, "OPT"), "OPT")
    normal_ids = sorted(set(sar_map) & set(opt_map))
    print(f"{len(normal_ids)} serie normali trovate", flush=True)
    baseline_ids = normal_ids[:n_normal_treshold]
    normal_test_ids = normal_ids[n_normal_treshold:]

    # dataset anomalo
    print("Download dataset anomalie", flush=True)
    anomalies_path = project_data.get_artifact("Floods_Anomalies").download("/data/anomalies_floods")

    def build_tar_map(tar_path, prefix):
        m = {}
        with tarfile.open(tar_path, 'r') as t:
            for n in t.getnames():
                if n.lower().endswith('.tif'):
                    ID = os.path.basename(n).split(f'_{prefix}_')[0]
                    m[ID] = tar_path
        return m

    sar_anom_map = build_tar_map(os.path.join(anomalies_path, "SAR_anomalies.tar"), "SAR")
    opt_anom_map = build_tar_map(os.path.join(anomalies_path, "OPT_anomalies.tar"), "OPT")
    anomaly_ids = sorted(set(sar_anom_map) & set(opt_anom_map))
    print(f"{len(anomaly_ids)} serie anomale trovate", flush=True)

    # carico pesi MoCo
    print("Carico i pesi MoCo", flush=True)
    moco_path = project_work.get_artifact(f"moco-weights_{moco_job_name}").download(f"moco_best_{moco_job_name}.pth")

    modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=10, n_images=n_images2, mamba=mamba).to(device)
    modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)

    model = MoCo2encoders(
        base_encoder_q=modelS2.encoder, base_encoder_k=modelS1.encoder,
        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t, symmetric=symmetric, device=device,
    ).to(device)
    model.load_state_dict(torch.load(moco_path, map_location=device))  
    model.eval()

    def make_loader(ids, s_map, o_map):
        ds = PairLoader(ids, s_map, o_map, patch_size, n_images1, n_channels1, n_images2, n_channels2)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    treshold_loader = make_loader(baseline_ids, sar_map, opt_map)
    normal_test_loader = make_loader(normal_test_ids, sar_map, opt_map)
    anomaly_loader = make_loader(anomaly_ids, sar_anom_map, opt_anom_map)

    cos = torch.nn.CosineSimilarity(dim=1)

    def compute_similarities(loader):
        sims = []
        with torch.no_grad():
            for im_q, im_k in tqdm(loader):
                im_q = im_q.to(non_blocking=True, device=device)
                im_k = im_k.to(non_blocking=True, device=device)
                _, q, k = model.contrastive_loss(im_q, im_k)
                sims.append(cos(q, k).cpu().numpy())
        return np.concatenate(sims, axis=0)

    # Calcolo soglia th_low e th_high
    print("Calcolo treshold", flush=True)
    baseline_sims = compute_similarities(treshold_loader)
    th_low = baseline_sims.mean() - baseline_sims.std()
    th_high = baseline_sims.mean() + baseline_sims.std()
    print(f"Th: [{th_low:.4f}, {th_high:.4f}]", flush=True)

    # Testing, dataset mix serie normali e anomale
    print("Testing: mix normali anomale", flush=True)
    normal_sims = compute_similarities(normal_test_loader)
    anomaly_sims = compute_similarities(anomaly_loader)

    predictions = np.concatenate([normal_sims, anomaly_sims])
    gt = np.concatenate([np.zeros(len(normal_sims)), np.ones(len(anomaly_sims))])
    detection = ((predictions < th_low) | (predictions > th_high)).astype(np.int8)

    # Metriche e risutati
    conf_matrix = confusion_matrix(gt, detection)
    precision = precision_score(gt, detection, zero_division=0)
    recall = recall_score(gt, detection, zero_division=0)
    f1 = f1_score(gt, detection, zero_division=0)
    
    print(f"Confusion matrix:\n{conf_matrix}", flush=True)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}", flush=True)

    np.save('confusion_matrix.npy', conf_matrix)
    np.save('similarities.npy', predictions)
    np.save('ground_truth.npy', gt)

    try:
        project_work.log_artifact(name="anomaly-confusion-matrix", source="confusion_matrix.npy")
        project_work.log_artifact(name="anomaly-similarities", source="similarities.npy")
        project_work.log_artifact(name="anomaly-ground-truth", source="ground_truth.npy")
    except Exception as e:
        print({e}, flush=True)

    return f"Anomaly detection completata — F1: {f1:.4f}"



@handler()
def detect_anomalies(
    n_normal_treshold: int = 500,                   # quante serie normali usare per calcolare la soglia
    patch_size: int = 256,
    n_images1: int = 4, n_channels1: int = 2,       # sar
    n_images2: int = 4, n_channels2: int = 10,      # ottico
    moco_dim: int = 128, moco_k: int = 1024, moco_m: float = 0.999, moco_t: float = 0.07,
    symmetric: bool = False, mamba: bool = False,
    batch_size: int = 8, workers: int = 0,
    moco_job_name: str = "moco_nome_job",
    dataset: str = "Test"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print('Using device:', device, flush=True)

    project_work = dh.get_project("floods")
    project_data = dh.get_project("datasets")

    # import dataset normale
    print("Download dataset normale", flush=True)
    dataset_path = project_data.get_artifact(f"Floods_{dataset}").download("/data/dataset_floods")

    def build_zip_map(base_dir, prefix):
        m = {}
        for zf in sorted(glob(os.path.join(base_dir, f"{prefix}_*.zip"))):
            with zipfile.ZipFile(zf, 'r') as z:
                for n in z.namelist():
                    if n.lower().endswith('.tif'):
                        ID = os.path.basename(n).split(f'_{prefix}_')[0]
                        m[ID] = zf
        return m

    sar_map = build_zip_map(os.path.join(dataset_path, "SAR"), "SAR")
    opt_map = build_zip_map(os.path.join(dataset_path, "OPT"), "OPT")
    normal_ids = sorted(set(sar_map) & set(opt_map))
    print(f"{len(normal_ids)} serie normali trovate", flush=True)
    baseline_ids = normal_ids[:n_normal_treshold]
    normal_test_ids = normal_ids[n_normal_treshold:]

    # dataset anomalo
    print("Download dataset anomalie", flush=True)
    anomalies_path = project_data.get_artifact("Floods_Anomalies").download("/data/anomalies_floods")

    def build_tar_map(tar_path, prefix):
        m = {}
        with tarfile.open(tar_path, 'r') as t:
            for n in t.getnames():
                if n.lower().endswith('.tif'):
                    ID = os.path.basename(n).split(f'_{prefix}_')[0]
                    m[ID] = tar_path
        return m

    sar_anom_map = build_tar_map(os.path.join(anomalies_path, "SAR_anomalies.tar"), "SAR")
    opt_anom_map = build_tar_map(os.path.join(anomalies_path, "OPT_anomalies.tar"), "OPT")
    anomaly_ids = sorted(set(sar_anom_map) & set(opt_anom_map))
    print(f"{len(anomaly_ids)} serie anomale trovate", flush=True)

    # carico pesi MoCo
    print("Carico i pesi MoCo", flush=True)
    moco_path = project_work.get_artifact(f"moco-weights_{moco_job_name}").download(f"moco_best_{moco_job_name}.pth")

    modelS2 = Singlemodal_CAE(input_dim=n_channels2, output_dim=10, n_images=n_images2, mamba=mamba).to(device)
    modelS1 = Singlemodal_CAE(input_dim=n_channels1, output_dim=10, n_images=n_images1, mamba=mamba).to(device)

    model = MoCo2encoders(
        base_encoder_q=modelS2.encoder, base_encoder_k=modelS1.encoder,
        dim=moco_dim, K=moco_k, m=moco_m, T=moco_t, symmetric=symmetric, device=device,
    ).to(device)
    model.load_state_dict(torch.load(moco_path, map_location=device))  
    model.eval()

    def make_loader(ids, s_map, o_map):
        ds = PairLoader(ids, s_map, o_map, patch_size, n_images1, n_channels1, n_images2, n_channels2)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    treshold_loader = make_loader(baseline_ids, sar_map, opt_map)
    normal_test_loader = make_loader(normal_test_ids, sar_map, opt_map)
    anomaly_loader = make_loader(anomaly_ids, sar_anom_map, opt_anom_map)

    cos = torch.nn.CosineSimilarity(dim=1)

    def compute_similarities(loader):
        sims = []
        with torch.no_grad():
            for im_q, im_k in tqdm(loader):
                im_q = im_q.to(non_blocking=True, device=device)
                im_k = im_k.to(non_blocking=True, device=device)
                _, q, k = model.contrastive_loss(im_q, im_k)
                sims.append(cos(q, k).cpu().numpy())
        return np.concatenate(sims, axis=0)

    # Calcolo soglia th_low e th_high
    print("Calcolo treshold", flush=True)
    baseline_sims = compute_similarities(treshold_loader)
    th_low = baseline_sims.mean() - baseline_sims.std()
    th_high = baseline_sims.mean() + baseline_sims.std()
    print(f"Th: [{th_low:.4f}, {th_high:.4f}]", flush=True)

    # Testing, dataset mix serie normali e anomale
    print("Testing: mix normali anomale", flush=True)
    normal_sims = compute_similarities(normal_test_loader)
    anomaly_sims = compute_similarities(anomaly_loader)

    predictions = np.concatenate([normal_sims, anomaly_sims])
    gt = np.concatenate([np.zeros(len(normal_sims)), np.ones(len(anomaly_sims))])
    detection = ((predictions < th_low) | (predictions > th_high)).astype(np.int8)

    # Metriche e risutati
    conf_matrix = confusion_matrix(gt, detection)
    precision = precision_score(gt, detection, zero_division=0)
    recall = recall_score(gt, detection, zero_division=0)
    f1 = f1_score(gt, detection, zero_division=0)
    
    print(f"Confusion matrix:\n{conf_matrix}", flush=True)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}", flush=True)

    np.save('confusion_matrix.npy', conf_matrix)
    np.save('similarities.npy', predictions)
    np.save('ground_truth.npy', gt)

    try:
        project_work.log_artifact(name="anomaly-confusion-matrix", source="confusion_matrix.npy")
        project_work.log_artifact(name="anomaly-similarities", source="similarities.npy")
        project_work.log_artifact(name="anomaly-ground-truth", source="ground_truth.npy")
    except Exception as e:
        print({e}, flush=True)

    return f"Anomaly detection completata — F1: {f1:.4f}"