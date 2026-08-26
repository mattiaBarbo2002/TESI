# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random

from PIL import ImageFilter
from torch.utils.data import Dataset
import tifffile as tiff
import os
import numpy as np
import rasterio
import zipfile


# lettura file zip senza estrarlo

def read_tiff_from_zip(zip_path, expected_tif_name):
    abs_zip_path = os.path.abspath(zip_path)
    vsi_path = f"/vsizip/{abs_zip_path}/{expected_tif_name}"

    with rasterio.open(vsi_path) as src:
        img_data = src.read()
        
    return np.nan_to_num(img_data, nan=0.0).astype(np.float32)


# normalizzazione valori

def normalize_image(img, data_type):
    if data_type == 'OPT':
        img = img / 10000.0
        img = np.clip(img, 0.0, 1.5)  

    else:  # 'SAR'
        img = np.clip(img, -25.0, 0.0)
        img = (img + 25.0) / 25.0  
    return img


# SINGOLO ENCODER

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
        
        print(f"Precalcolo {len(listIDs)} serie {data_type}", flush=True)
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
  


# CLASSE MOCO
# stessa logica Singlemodal_Loader

class MoCo2encodersLoader(Dataset):
    def __init__(self, listIDs, root, transform, patch_size=256,
                 n_images1=4, n_channels1=2, n_images2=4, n_channels2=10):
        self.listIDs = listIDs
        self.root = root
        self.transform = transform
        self.patch_size = patch_size
        self.n_images1 = n_images1              # sar
        self.n_channels1 = n_channels1
        self.n_images2 = n_images2              # ottico
        self.n_channels2 = n_channels2

        self.cache_dir_sar = "/data/cache_moco_SAR"
        self.cache_dir_opt = "/data/cache_moco_OPT"
        os.makedirs(self.cache_dir_sar, exist_ok=True)
        os.makedirs(self.cache_dir_opt, exist_ok=True)

        print(f"Precalcolo {len(listIDs)} serie SAR", flush=True)
        for ID in self.listIDs:
            save_path = os.path.join(self.cache_dir_sar, f"{ID}.npy")
            if not os.path.exists(save_path):
                im = self._load_and_process(ID, self.sar_map, 'SAR', self.n_images1, self.n_channels1)
                np.save(save_path, im)
        print("Precalcolo SAR completato", flush=True)

        for zip_path in set(self.sar_map.values()):
            if os.path.exists(zip_path):
                os.remove(zip_path)
        print("OK -> ZIP SAR eliminati", flush=True)

        print(f"Precalcolo {len(listIDs)} serie OPT", flush=True)
        for ID in self.listIDs:
            save_path = os.path.join(self.cache_dir_opt, f"{ID}.npy")
            if not os.path.exists(save_path):
                im = self._load_and_process(ID, self.opt_map, 'OPT', self.n_images2, self.n_channels2)
                np.save(save_path, im)
        print("Precalcolo OPT completato", flush=True)

        for zip_path in set(self.opt_map.values()):
            if os.path.exists(zip_path):
                os.remove(zip_path)
        print("OK -> ZIP OPT eliminati", flush=True)

    def _load_and_process(self, ID, archive_map, suffix, n_images, n_channels):
        im = np.zeros((n_channels, n_images, self.patch_size, self.patch_size), dtype=np.float32)
        serie_temporale = []
        zip_path = archive_map[ID]

        for t in range(n_images):
            time_step = t + 1
            base_name = f"{ID}_{suffix}_t{time_step}"
            img_data = read_tiff_from_zip(zip_path, f"{base_name}.tif")
            img_data = normalize_image(img_data, suffix)
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

        im_sar = np.load(os.path.join(self.cache_dir_sar, f"{ID}.npy"), mmap_mode='r')
        im_sar = torch.from_numpy(im_sar.copy())

        im_opt = np.load(os.path.join(self.cache_dir_opt, f"{ID}.npy"), mmap_mode='r')
        im_opt = torch.from_numpy(im_opt.copy())

        if self.transform is not None:
            im_q = self.transform(im_opt)
            im_k = self.transform(im_sar)
        else:
            im_q = im_opt
            im_k = im_sar

        return im_q, im_k

    def __len__(self):
        return len(self.listIDs)



# CLASSI E FUNZIONI ORIGINALI  

"""
class MoCo2encodersLoader(Dataset):
    def __init__(self, listIDs, root, transform, patch_size=256,
                 n_images1=4, n_channels1=3, n_images2=4, n_channels2=2):
        self.listIDs = listIDs
        self.root = root
        self.transform = transform
        self.patch_size = patch_size
        self.n_images1 = n_images1              # sar
        self.n_channels1 = n_channels1
        self.n_images2 = n_images2              # ottico
        self.n_channels2 = n_channels2

    def __getitem__(self, index):
        ID = self.listIDs[index]
        
        # tensori vuoti
        im_1 = np.empty((self.n_channels1, self.n_images1, self.patch_size, self.patch_size), dtype=np.float32)
        im_2 = np.empty((self.n_channels2, self.n_images2, self.patch_size, self.patch_size), dtype=np.float32)
        
        # caricamento img SAR
        for t in range(self.n_images1):
            time_step = t + 1
            zip_path_s1 = os.path.join(self.root, 'SAR', f"{ID}_SAR_t{time_step}.zip")
            
            img_s1 = read_tiff_from_zip(zip_path_s1)
            c_to_take = min(self.n_channels1, img_s1.shape[0])
            im_1[:c_to_take, t, :, :] = img_s1[:c_to_take, :, :]
                    
        # caricamento img OPT
        for t in range(self.n_images2):
            time_step = t + 1
            zip_path_s2 = os.path.join(self.root, 'OPT', f"{ID}_OPT_t{time_step}.zip")
            
            img_s2 = read_tiff_from_zip(zip_path_s2)
            c_to_take = min(self.n_channels2, img_s2.shape[0])
            im_2[:c_to_take, t, :, :] = img_s2[:c_to_take, :, :]
            print("read_tiff_from_zip S2")

        # coversione in tensori
        im_1 = torch.from_numpy(im_1)
        im_2 = torch.from_numpy(im_2)

        if self.transform is not None:
            im_q = self.transform(im_2)
            im_k = self.transform(im_1)
        else:
            im_q = im_2
            im_k = im_1

        return im_q, im_k

    def __len__(self):
        return len(self.listIDs)

"""    

# ALTRE CLASSI E FUNZIONI MAI USATE

"""
class MoCoLoader(Dataset):
    def __init__(self, listIDs, root, transform, patch_size=256, 
    n_images1=1, n_channels1=1, n_images2=1, n_channels2=1) -> None:
        self.listIDs = listIDs
        self.root = root
        self.transform = transform
        self.patch_size = patch_size
        self.n_images1 = n_images1
        self.n_channels1 = n_channels1
        self.n_images2 = n_images2
        self.n_channels2 = n_channels2

    # prende i dati .tiff dal dataset (unici per sar e ottico)
    # da modificare con mio dataset
    def __getitem__(self, index):

        ID = self.listIDs[index]
        #print(os.path.join(self.root, ID))
        img = np.array(tiff.imread(os.path.join(self.root, ID))).astype(np.float32)
        img[np.isnan(img)] = 0
        im_1 = np.empty((self.n_channels1, self.n_images1, self.patch_size,self.patch_size), dtype=np.float32)
        im_2 = np.empty((self.n_channels2, self.n_images2, self.patch_size,self.patch_size), dtype=np.float32)
        for i_x in range(20):
            time_index = int(np.floor(i_x/2))
            channel_index = int(i_x%2)
            im_1[channel_index, time_index, :, :] = img[128:384,128:384, i_x]
        im_2[0,:,:,:] = np.transpose(img[128:384,128:384,20:20+self.n_images2],(2,0,1))

        im_1 = torch.from_numpy(im_1)#.long()
        im_2 = torch.from_numpy(im_2)#.long()

        if self.transform is not None:
            im_q1 = self.transform(im_1)
            im_q2 = self.transform(im_2)
            im_k1 = self.transform(im_1)
            im_k2 = self.transform(im_2)
        else:
            im_q1 = im_1
            im_q2 = im_2
            im_k1 = im_1
            im_k2 = im_2

        # crea i dizioniari, vedi builder riga 309
        im_q = {'im1': im_q1, 'im2': im_q2}
        im_k = {'im1': im_k1, 'im2': im_k2}
        return im_q, im_k

    def __len__(self):
        return len(self.listIDs)

class MoCo2encodersLoader(Dataset):
    def __init__(self, listIDs, root, transform, patch_size=256, 
    n_images1=1, n_channels1=1, n_images2=1, n_channels2=1) -> None:
        self.listIDs = listIDs
        self.root = root
        self.transform = transform
        self.patch_size = patch_size
        self.n_images1 = n_images1
        self.n_channels1 = n_channels1
        self.n_images2 = n_images2
        self.n_channels2 = n_channels2

    def __getitem__(self, index):
        ID = self.listIDs[index]
        #print(os.path.join(self.root, ID))
        img = np.array(tiff.imread(os.path.join(self.root, ID))).astype(np.float32)
        img[np.isnan(img)] = 0
        im_1 = np.empty((self.n_channels1, self.n_images1, self.patch_size,self.patch_size), dtype=np.float32)
        im_2 = np.empty((self.n_channels2, self.n_images2, self.patch_size,self.patch_size), dtype=np.float32)
        for i_x in range(20):
            time_index = int(np.floor(i_x/2))
            channel_index = int(i_x%2)
            im_1[channel_index, time_index, :, :] = img[128:384,128:384, i_x]
        im_2[0,:,:,:] = np.transpose(img[128:384,128:384,20:20+self.n_images2],(2,0,1))

        im_1 = torch.from_numpy(im_1)#.long()
        im_2 = torch.from_numpy(im_2)#.long()

        if self.transform is not None:
            im_q = self.transform(im_1)
            im_k = self.transform(im_2)
        else:
            im_q = im_1
            im_k = im_2
        # im_q = {'im1': im_q1, 'im2': im_q2}
        # im_k = {'im1': im_k1, 'im2': im_k2}
        return im_q, im_k

    def __len__(self):
        return len(self.listIDs)

class Singlemodal_Loader(Dataset):
    def __init__(self, listIDs, root, transform, patch_size=256, 
    n_images=1, n_channels=1, data_type='S1') -> None:
        self.listIDs = listIDs
        self.root = root
        self.transform = transform
        self.patch_size = patch_size
        self.n_images = n_images
        self.n_channels = n_channels
        self.data_type = data_type

    def __getitem__(self, index):
        ID = self.listIDs[index]
        #print(os.path.join(self.root, ID))
        img = np.array(tiff.imread(os.path.join(self.root, ID))).astype(np.float32)
        img[np.isnan(img)] = 0
        im = np.empty((self.n_channels, self.n_images, self.patch_size,self.patch_size), dtype=np.float32)
        if self.data_type=='S2':
            for i_x in range(20):
                time_index = int(np.floor(i_x/2))
                channel_index = int(i_x%2)
                im[channel_index, time_index, :, :] = img[128:384,128:384, i_x]
        else:
            im[0,:,:,:] = np.transpose(img[128:384,128:384,20:20+self.n_images],(2,0,1))

        im = torch.from_numpy(im)#.long()

        if self.transform is not None:
            im = self.transform(im)
            
        return im

    def __len__(self):
        return len(self.listIDs)

# funzioni augmentation, probabilemente estratte da paper

class TwoCropsTransform:
    # Take two random crops of one image as the query and key.

    def __init__(self, base_transform) -> None:
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    # Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709

    def __init__(self, sigma=[0.1, 2.0]) -> None:
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
"""    