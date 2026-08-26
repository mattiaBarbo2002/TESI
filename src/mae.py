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
#from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


import time
import zipfile
from glob import glob

import torchvision.transforms as transforms
from digitalhub_runtime_python import handler

from multimodal_3Dconv_attention import Singlemodal_CAE
from moco.loader import Singlemodal_Loader

# S2 -> encoder mascherato -> decoder -> costruisce S1
# o il contrario

# aggiungere maschera encoder
# modifica output decoder -> passare coppia di serie non singola modalità
# passare serie S1 a decoder per calcolo loss
# training contemporaneo o separato?

class Singlemodal_Loader(Dataset):
    def __init__(self, listIDs, root, zip_map, transform, patch_size=256,
                 n_images=4, n_channels=3, data_type='SAR'):
        self.listIDs = listIDs
        self.root = root
        self.zip_map = zip_map
        self.transform = transform
        self.patch_size = patch_size
        self.n_images = n_images
        self.n_channels = n_channels
        self.data_type = data_type

        
        suffix = 'OPT' if self.data_type == 'OPT' else 'SAR'
        self.cache_dir = f"/data/cache_{suffix}"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Precalcolo {len(listIDs)} serie {data_type} su disco...", flush=True)
        for ID in self.listIDs:
            save_path = os.path.join(self.cache_dir, f"{ID}.npy")
            # Calcola e salva solo se non esiste già
            if not os.path.exists(save_path):
                im = self._load_and_process(ID, suffix)
                np.save(save_path, im)
        print(f"Precalcolo {data_type} completato", flush=True)

    def _load_and_process(self, ID, suffix):
        im = np.zeros((self.n_channels, self.n_images, self.patch_size, self.patch_size), dtype=np.float32)
        serie_temporale = []

        zip_path = self.zip_map[ID]   # lo zip specifico (una delle 6 parti) che contiene questa serie

        for t in range(self.n_images):
            time_step = t + 1
            base_name = f"{ID}_{suffix}_t{time_step}"
            img_data = read_tiff_from_zip(zip_path, f"{base_name}.tif")   
            img_data = normalize_image(img_data, self.data_type)
            serie_temporale.append(img_data)

        c_tot, h_tot, w_tot = serie_temporale[0].shape
        start_y = max(0, (h_tot - self.patch_size) // 2)
        start_x = max(0, (w_tot - self.patch_size) // 2)
        pad_y = max(0, self.patch_size - h_tot)
        pad_x = max(0, self.patch_size - w_tot)

        for t in range(self.n_images):
            img = serie_temporale[t]
            c_to_take = min(self.n_channels, img.shape[0])
            img_cropped = img[:c_to_take, start_y:start_y+self.patch_size, start_x:start_x+self.patch_size]
            if pad_y > 0 or pad_x > 0:
                img_cropped = np.pad(img_cropped, ((0,0), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0.0)
            im[:c_to_take, t, :, :] = img_cropped

        return im    


    def __getitem__(self, index):
        ID = self.listIDs[index]
        save_path = os.path.join(self.cache_dir, f"{ID}.npy")

        im = np.load(save_path, mmap_mode='r')
        im = torch.from_numpy(im.copy())

        if self.transform is not None:
            im = self.transform(im)
        return im

    def __len__(self):
        return len(self.listIDs)