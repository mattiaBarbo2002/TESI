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


# --- CLASSE MOCO ---

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


# --- SINGOLO ENCODER ---

class Singlemodal_Loader(Dataset):
    def __init__(self, listIDs, root, transform, patch_size=256,
                 n_images=4, n_channels=3, data_type='SAR'):
        self.listIDs = listIDs
        self.root = root
        self.transform = transform
        self.patch_size = patch_size
        self.n_images = n_images
        self.n_channels = n_channels
        self.data_type = data_type 

    def __getitem__(self, index):
        ID = self.listIDs[index]
        
        im = np.zeros((self.n_channels, self.n_images, self.patch_size, self.patch_size), dtype=np.float32)
        
        folder = 'OPT' if self.data_type == 'OPT' else 'SAR'
        suffix = 'OPT' if self.data_type == 'OPT' else 'SAR'
        
        serie_temporale = []
        
        # lettura immagini singola serie temporale
        for t in range(self.n_images):
            time_step = t + 1
            base_name = f"{ID}_{suffix}_t{time_step}"
            zip_path = os.path.join(self.root, folder, f"{base_name}.zip")
            tif_name = f"{base_name}.tif"
            
            img_data = read_tiff_from_zip(zip_path, tif_name)

            # normalizzazione 
            img_data = normalize_image(img_data, self.data_type)   
            serie_temporale.append(img_data)
            
        # crop da t=1
        c_tot, h_tot, w_tot = serie_temporale[0].shape
        
        start_y = max(0, (h_tot - self.patch_size) // 2)
        start_x = max(0, (w_tot - self.patch_size) // 2)
        
        pad_y = max(0, self.patch_size - h_tot)
        pad_x = max(0, self.patch_size - w_tot)

        # stesso taglio sugli altri t
        for t in range(self.n_images):
            img = serie_temporale[t]
            c_to_take = min(self.n_channels, img.shape[0])
            
            img_cropped = img[:c_to_take, start_y:start_y+self.patch_size, start_x:start_x+self.patch_size]
            
            if pad_y > 0 or pad_x > 0:
                img_cropped = np.pad(img_cropped, ((0,0), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0.0)
                
            # img nel tensore
            im[:c_to_take, t, :, :] = img_cropped

        im = torch.from_numpy(im)
        if self.transform is not None:
            im = self.transform(im)
            
        return im

    def __len__(self):
        return len(self.listIDs)        
      


# DOMANDE 
# 1. augmentation

# ALTRE FUNZIONI ORIGINALI

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