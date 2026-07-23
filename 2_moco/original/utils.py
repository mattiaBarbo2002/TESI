# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:00:36 2019

@author: lbergamasco
"""

import numpy as np
import os,sys
from os.path import dirname
import gc
sys.path.append(dirname(__file__))

def list_files(directory, extension):
    list = []
    for f in os.listdir(directory): 
        if f.endswith(extension):
            list.append(f)
    return list

def list_dir(directory):
    list = []
    for f in os.listdir(directory): 
        if os.path.isdir(directory+f):
            list.append(f)
    return list

def merge_dataset(path, output_filename):
    list_dataset = list_files(path,'.npy')
    dataset = None
    for d in list_dataset:
        if dataset is None:
            dataset = np.load(path + '/' + d)
        else:
            dataset = np.concatenate([dataset,np.load(path + '/' + d)], axis = 0)
            gc.collect()
    
    np.save(path + '/' + output_filename + '.npy', dataset)
