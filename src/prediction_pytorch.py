#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:53:21 2023

@author: lbergamasco
"""

# ANOMALY DETECTION -DA MODIFICARE

import os
import torch
from src.multimodal_3Dconv_attention import Singlemodal_Encoder, Multimodal_Encoder, Singlemodal_CAE
import src.moco.loader
import src.moco.builder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tifffile as tiff
from scipy.io import savemat

def main():
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch MoCo Training')
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="multimodal_encoder",
        choices=["multimodal_encoder"],
        help="model architecture: " + " | singlemodal_encoder | multimodal_encoder (default: multimodal_encoder)",
    )
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--symmetric', action='store_true',
                        help='use symmetric loss')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    args = parser.parse_args()
    
    print("args:", args)

    models = {
        "singlemodal_encoder": Singlemodal_Encoder,
        "multimodal_encoder": Multimodal_Encoder}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    ROW_AXIS = 1
    COLUMN_AXIS = 2
    CHANNEL_AXIS = 3

    model_filename = 'Experiments/MoCoMultimodal_pretrained_Mamba/model_best.pth'

    dataset_path = 'Datasets/Norway/Big_lakes_norm/'
        
    draining_images_path = dataset_path+'draining/'
    non_draining_images_path = dataset_path+'non-draining/'

    output_path = 'Experiments/MoCoMultimodal_pretrained_Mamba/predictions/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    batch_size = 8
    patch_size = 256
    n_images1 = 4
    n_images2 = 4
    #n_images3 = 21
    n_channels1 = 2
    n_channels2 = 1
    #n_channels3 = 1
    image_size1 = (patch_size,patch_size,n_images1,n_channels1)
    #image_size2 = (1,patch_size,patch_size,n_images2,n_channels2+1)
    #image_size3 = (1,patch_size,patch_size,n_images3,n_channels3+1)

    modelS2 = Singlemodal_CAE(input_dim= n_channels1, output_dim=10, n_images= n_images1, mamba=True).to(device)
    modelS1 = Singlemodal_CAE(input_dim= n_channels2, output_dim=10, n_images= n_images2, mamba=True).to(device)

    # model = moco.builder.MoCo(
    #         models[args.arch],
    #         args.moco_dim,
    #         args.moco_k,
    #         args.moco_m,
    #         args.moco_t,
    #         args.mlp,
    #         symmetric=args.symmetric,
    #         device=device,
    #     ).to(device=device)

    model = notebooks.src.moco.builder.MoCo2encoders(
        base_encoder_q = modelS2.encoder,
        base_encoder_k = modelS1.encoder,
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        symmetric=args.symmetric,
        device=device,
    ).to(device=device)
    print(model)
    
    model.load_state_dict(torch.load(model_filename,map_location=device)['state_dict'])
    model.eval()

    list_non_draining_training = [non_draining_images_path+'train/'+f for f in os.listdir(non_draining_images_path+'train/')]

    list_non_draining_files = [non_draining_images_path+'val/'+f for f in os.listdir(non_draining_images_path+'val/')]

    list_draining_files = [draining_images_path+'train/'+f for f in os.listdir(draining_images_path+'train/')]
    list_draining_files.extend([draining_images_path+'val/'+f for f in os.listdir(draining_images_path+'val/')])
    list_test_files = list_non_draining_files+list_draining_files
    dim1 = image_size1[:2]

    non_draining_training_dataset = notebooks.src.moco.loader.MoCo2encodersLoader(
        listIDs = list_non_draining_training,
        root = './',
        transform = None,
        patch_size = patch_size,
        n_images1 = n_images1,
        n_channels1 = n_channels1,
        n_images2 = n_images2,
        n_channels2 = n_channels2
    )

    draining_dataset = notebooks.src.moco.loader.MoCo2encodersLoader(
        listIDs = list_draining_files,
        root = './',
        transform = None,
        patch_size = patch_size,
        n_images1 = n_images1,
        n_channels1 = n_channels1,
        n_images2 = n_images2,
        n_channels2 = n_channels2
    )

    non_draining_dataset = notebooks.src.moco.loader.MoCo2encodersLoader(
        listIDs = list_non_draining_files,
        root = './',
        transform = None,
        patch_size = patch_size,
        n_images1 = n_images1,
        n_channels1 = n_channels1,
        n_images2 = n_images2,
        n_channels2 = n_channels2
    )

    non_draining_training_loader = torch.utils.data.DataLoader(
        non_draining_training_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    
    draining_loader = torch.utils.data.DataLoader(
        draining_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    non_draining_loader = torch.utils.data.DataLoader(
        non_draining_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    non_draining_training_predictions = []
    non_draining_predictions = []
    draining_predictions = []

    for im_1, im_2 in non_draining_training_loader:
        im_1, im_2 = im_1.to(non_blocking=True, device=device), im_2.to(non_blocking=True, device=device)
        with torch.no_grad():
            _,output1, output2 = model.contrastive_loss(im_1,im_2)#.encoder_q(im_1,im_2)
            output = torch.nn.CosineSimilarity(dim=1)(output1,output2)
            output = output.cpu().numpy()
            non_draining_training_predictions.append(output)
    non_draining_training_predictions = np.concatenate(non_draining_training_predictions,axis=0)
    th1 = np.mean(non_draining_training_predictions,axis=0)+np.std(non_draining_training_predictions,axis=0)
    th2 = np.mean(non_draining_training_predictions,axis=0)-np.std(non_draining_training_predictions,axis=0)
    print('Thresholds for normal data: {} - {}'.format(th1,th2))

    input_images_non_draining_S2 = []
    input_images_non_draining_S1 = []
    output_images_non_draining_S2 = []
    output_images_non_draining_S1 = []
    for im_1, im_2 in non_draining_loader:
        im_1, im_2 = im_1.to(non_blocking=True, device=device), im_2.to(non_blocking=True, device=device)
        with torch.no_grad():
            _,output1, output2 = model.contrastive_loss(im_1,im_2)#output = model.encoder_q(im_1)
            output = torch.nn.CosineSimilarity(dim=1)(output1,output2)
            output = output.cpu().numpy()
            non_draining_predictions.append(output)
            output_images_non_draining_S2.append(output1.cpu().numpy())
            output_images_non_draining_S1.append(output2.cpu().numpy())
        input_images_non_draining_S2.append(im_1.cpu().numpy())
        input_images_non_draining_S1.append(im_2.cpu().numpy())
    non_draining_predictions = np.concatenate(non_draining_predictions,axis=0)
    input_images_non_draining_S2 = np.concatenate(input_images_non_draining_S2,axis=0)
    input_images_non_draining_S1 = np.concatenate(input_images_non_draining_S1,axis=0)
    output_images_non_draining_S2 = np.concatenate(output_images_non_draining_S2,axis=0)
    output_images_non_draining_S1 = np.concatenate(output_images_non_draining_S1,axis=0)

    input_images_draining_S2 = []
    input_images_draining_S1 = []
    output_images_draining_S2 = []
    output_images_draining_S1 = []
    for im_1, im_2 in draining_loader:
        im_1, im_2 = im_1.to(non_blocking=True, device=device), im_2.to(non_blocking=True, device=device)
        with torch.no_grad():
            _,output1, output2 = model.contrastive_loss(im_1,im_2)#output = model.encoder_q(im_1)
            output = torch.nn.CosineSimilarity(dim=1)(output1,output2)
            output = output.cpu().numpy()
            draining_predictions.append(output)
            output_images_draining_S2.append(output1.cpu().numpy())
            output_images_draining_S1.append(output2.cpu().numpy())
        input_images_draining_S2.append(im_1.cpu().numpy())
        input_images_draining_S1.append(im_2.cpu().numpy())
    draining_predictions = np.concatenate(draining_predictions,axis=0)
    input_images_draining_S2 = np.concatenate(input_images_draining_S2,axis=0)
    input_images_draining_S1 = np.concatenate(input_images_draining_S1,axis=0)
    output_images_draining_S2 = np.concatenate(output_images_draining_S2,axis=0)
    output_images_draining_S1 = np.concatenate(output_images_draining_S1,axis=0)

    print(non_draining_predictions.shape)
    print(draining_predictions.shape)

    predictions = np.concatenate([non_draining_predictions,draining_predictions],axis=0)
    input_images_S2 = np.concatenate([input_images_non_draining_S2,input_images_draining_S2],axis=0)
    input_images_S1 = np.concatenate([input_images_non_draining_S1,input_images_draining_S1],axis=0)
    output_images_S2 = np.concatenate([output_images_non_draining_S2,output_images_draining_S2],axis=0)
    output_images_S1 = np.concatenate([output_images_non_draining_S1,output_images_draining_S1],axis=0)

    gt = np.concatenate([np.zeros(len(list_non_draining_files)),np.ones(len(list_draining_files))],axis = 0)
    print('Prediction shape: {}'.format(predictions.shape))
    print('GT shape: {}'.format(gt.shape))

    # plt.plot(th1,label='Upper threshold')
    # plt.plot(th2,label='Lower threshold')
    # plt.plot(predictions[0,:],label='Non-draining example')
    # plt.plot(predictions[-1,:],label='Draining example')
    # plt.legend()
    # plt.title('Anomaly detection thresholds')
    # plt.savefig(output_path+'thresholds.png')
    plt.hist(predictions[gt==0],bins=100,label='Non-draining')
    plt.hist(predictions[gt==1],bins=100,label='Draining')
    plt.axvline(x=th1,color='r',label='Upper threshold')
    plt.axvline(x=th2,color='g',label='Lower threshold')
    plt.legend()
    plt.title('Prediction distribution')
    plt.savefig(output_path+'prediction_distribution.png')

    detection = np.int8(predictions>th1)#np.int8(np.logical_or(predictions>th1,predictions<th2).any(axis=1))

    conf_matrix = confusion_matrix(gt,detection)
    print('Confusion matrix:\n{}'.format(conf_matrix))
    np.save(output_path+'conf_matrix.npy',conf_matrix)
    print('FA rate: {}'.format(conf_matrix[0,1]/conf_matrix[0,:].sum()))
    print('MA rate: {}'.format(conf_matrix[1,0]/conf_matrix[1,:].sum()))
    print('OE: {}'.format((conf_matrix[0,1]+conf_matrix[1,0])/np.sum(conf_matrix)))
    print('Recall: {}'.format(recall_score(gt,detection,average=None)[1]))
    print('Precision: {}'.format(precision_score(gt,detection,average=None)[1]))
    f1 = f1_score(gt,detection,average=None)
    print('F1-score\nNot changed: {}\nChanged: {}'.format(f1[0],f1[1]))

    mdic = {"input_images_S2": input_images_S2,
            "input_images_S1": input_images_S1,
            "output_images_S2": output_images_S2,
            "output_images_S1": output_images_S1,
            "cosine_similarity_output": predictions,
            "gt": gt,
            "detection": detection}
    savemat(output_path+'draining_detection_results.mat', mdic)

if __name__ == '__main__':
    main()
    