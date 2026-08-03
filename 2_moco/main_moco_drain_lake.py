    #!/usr/bin/env python

    # pyre-unsafe

    # Copyright (c) Meta Platforms, Inc. and affiliates.

    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory of this source tree.
    # python main_moco_drain_lake.py --gpu 0 --moco-dim 10 --results-dir Experiments/MoCoMultimodal_pretrained --batch-size 8 --lr 0.0001 Datasets/Norway/Big_lakes_norm/non-draining/

    import argparse
    import builtins
    import math
    import os
    import random
    import shutil
    import time
    import warnings

    import moco.builder
    import moco.loader
    import torch
    import torch.backends.cudnn as cudnn
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.nn.parallel
    import torch.optim
    import torch.utils.data
    import torch.utils.data.distributed
    import torchvision.datasets as datasets
    from multimodal_3Dconv_attention import Singlemodal_Encoder, Multimodal_Encoder, Singlemodal_CAE
    import torchvision.transforms as transforms
    from tqdm import tqdm
    import pandas as pd
    from datetime import datetime
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, "\n")

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("data", default="Datasets/Norway/Big_lakes_norm/non-draining", metavar=" ", help="path to dataset")
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="multimodal_encoder_pretrained",
        choices=["singlemodal_encoder","multimodal_encoder","multimodal_encoder_pretrained"],
        help="model architecture: " + " | singlemodal_encoder | multimodal_encoder | multimodal_encoder_pretrained(default: multimodal_encoder_pretrained)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=16,
        type=int,
        metavar="N",
        help="mini-batch size (default: 16), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.03,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--schedule",
        default=[120, 160],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    # moco specific configs:
    parser.add_argument(
        "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
    )
    parser.add_argument(
        "--moco-k",
        default=65536,
        type=int,
        help="queue size; number of negative keys (default: 65536)",
    )
    parser.add_argument(
        "--moco-m",
        default=0.999,
        type=float,
        help="moco momentum of updating key encoder (default: 0.999)",
    )
    parser.add_argument(
        "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
    )

    # options for moco v2
    parser.add_argument("--mlp", action="store_true", help="use mlp head")
    parser.add_argument(
        "--aug-plus", action="store_true", help="use moco v2 data augmentation"
    )
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")
    parser.add_argument("--mmsingle", action="store_true", help="use multimodal input for one branch model")
    parser.add_argument("--patch_size", default=256, type=int, help="patch size")
    parser.add_argument("--n_images1", default=10, type=int, help="number of images in modality 1")
    parser.add_argument("--n_channels1", default=2, type=int, help="number of channels in modality 1")
    parser.add_argument("--n_images2", default=20, type=int, help="number of images in modality 2")
    parser.add_argument("--n_channels2", default=1, type=int, help="number of channels in modality 2")
    parser.add_argument('--symmetric', default=False, action='store_true',
                        help='use a symmetric loss function that backprops to both crops')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument('--resume-pretrainS1', default='', type=str, metavar='PATH', help='path to latest checkpoint of S1 CAE (default: none)')
    parser.add_argument('--resume-pretrainS2', default='', type=str, metavar='PATH', help='path to latest checkpoint of S2 CAE (default: none)')
    parser.add_argument('--use_pretrained-S1', default=False, action='store_true', help='use pretrained weights for S1 encoder')
    parser.add_argument('--use_pretrained-S2', default=False, action='store_true', help='use pretrained weights for S2 encoder')
    parser.add_argument('--mamba', default=False, action='store_true', help='use Mamba transformer module in CAE encoders')

    def main() -> None:
        args = parser.parse_args()
        if args.results_dir == '':
            args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        # Data loading code
        traindir = os.path.join(args.data, "train")
        train_data_IDS = os.listdir(traindir)

        augmentation = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=[-90,90], expand=False, fill=0),
            #transforms.ToTensor(),
            
        ]

        if args.use_pretrained_S2:
            if os.path.isfile(args.results_dir + '/modelS2_best.pth'):
                print("=> using pretrained weights for S2 encoder '{}'".format(args.results_dir + '/modelS2_best.pth'))
                args.resume_pretrainS2 = args.results_dir + '/modelS2_best.pth'
                modelS2 = Singlemodal_CAE(input_dim= args.n_channels1, output_dim=10, n_images= args.n_images1,mamba= args.mamba).to(device)
                if args.gpu is None:
                    checkpoint = torch.load(args.resume_pretrainS2)
                else:
                    loc = "cuda:{}".format(args.gpu)
                    checkpoint = torch.load(args.resume_pretrainS2, map_location=loc)
                modelS2.load_state_dict(checkpoint['state_dict'])
                print("=> loaded pretrained weights for S2 encoder '{}'".format(args.results_dir + '/modelS2_best.pth'))
            else:
                print("=> no pretrained weights found at '{}'".format(args.results_dir + '/modelS2_best.pth'))
        else:
            # create CAE for the S2 modality and pretrain it
            print("\nCreating and pretraining CAE for S2 modality...")
            modelS2 = Singlemodal_CAE(input_dim= args.n_channels1, output_dim=10, n_images= args.n_images1, mamba= args.mamba).to(device)
            optimizerS2 = torch.optim.Adam(modelS2.parameters(), lr=1e-4, weight_decay=args.weight_decay)
            loss_fn = nn.MSELoss().to(device)
            scalerS2 = torch.cuda.amp.GradScaler()

            best_loss = 9999.0
            if args.resume_pretrainS2 != '':
                print("=> loading checkpoint of S2 CAE '{}'".format(args.resume_pretrainS2))
                if os.path.isfile(args.resume_pretrainS2):
                    if args.gpu is None:
                        checkpoint = torch.load(args.resume_pretrainS2)
                    else:
                        loc = "cuda:{}".format(args.gpu)
                        checkpoint = torch.load(args.resume_pretrainS2, map_location=loc)
                    args.start_epoch = checkpoint['epoch']
                    modelS2.load_state_dict(checkpoint['state_dict'])
                    optimizerS2.load_state_dict(checkpoint['optimizer'])
                    best_loss = checkpoint['loss']
                    print(
                        "=> loaded checkpoint '{}' (epoch {})".format(
                            args.resume, checkpoint["epoch"]
                        )
                    )
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume_pretrainS2))

            datasetS2 = moco.loader.Singlemodal_Loader(
                listIDs = train_data_IDS,
                root = traindir,
                transform = transforms.Compose(augmentation),
                patch_size = args.patch_size,
                n_images = args.n_images1,
                n_channels = args.n_channels1,
                data_type = 'S2'
            )
            dataloaderS2 = torch.utils.data.DataLoader(
                datasetS2,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True,
            )
            resultsS2 = {'lr': [], 'train_loss': []}

            # pretrain CAE for modality 1 (S2)
            for epoch in range(1, args.epochs + 1):
                modelS2.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS2)
                for im in train_bar:
                    im = im.to(non_blocking=True, device=device)
                    # forward
                    output = modelS2(im)
                    print(im.dtype, output.dtype)  # debug dtype
                    loss = loss_fn(output, im)
                    # backward
                    optimizerS2.zero_grad()
                    scalerS2.scale(loss).backward()
                    scalerS2.step(optimizerS2)
                    scalerS2.update()
                    total_num += dataloaderS2.batch_size
                    total_loss += loss.item() * dataloaderS2.batch_size
                    train_bar.set_description(
                        'Pretrain S2 CAE Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, args.epochs,
                                                                            loss.item()))
                print("Pretrain S2 CAE - epoch total_loss/total_num", total_loss / total_num)
                resultsS2['lr'].append(optimizerS2.param_groups[0]['lr'])
                resultsS2['train_loss'].append(total_loss / total_num)
                # save statistics
                data_frame = pd.DataFrame(data=resultsS2, index=range(1, epoch + 1))
                data_frame.to_csv(args.results_dir + '/log_pretrainS2.csv', index_label='epoch')
                # save model
                torch.save({'epoch': epoch, 'state_dict': modelS2.state_dict(), 'optimizer': optimizerS2.state_dict(), 'loss': total_loss / total_num},
                        args.results_dir + '/modelS2_last.pth')
                # save best model
                if total_loss / total_num < best_loss:
                    shutil.copyfile(args.results_dir + '/modelS2_last.pth', args.results_dir + '/modelS2_best.pth')
                    best_loss = total_loss / total_num
            del modelS2

        torch.cuda.empty_cache()
        if args.use_pretrained_S1:
            if os.path.isfile(args.results_dir + '/modelS1_best.pth'):
                print("=> using pretrained weights for S1 encoder '{}'".format(args.results_dir + '/modelS1_best.pth'))
                args.resume_pretrainS1 = args.results_dir + '/modelS1_best.pth'
                modelS1 = Singlemodal_CAE(input_dim= args.n_channels2, output_dim=10, n_images= args.n_images2, mamba= args.mamba).to(device)
                if args.gpu is None:
                    checkpoint = torch.load(args.resume_pretrainS1)
                else:
                    loc = "cuda:{}".format(args.gpu)
                    checkpoint = torch.load(args.resume_pretrainS1, map_location=loc)
                modelS1.load_state_dict(checkpoint['state_dict'])
                print("=> loaded pretrained weights for S1 encoder '{}'".format(args.results_dir + '/modelS1_best.pth'))
            else:
                print("=> no pretrained weights found at '{}'".format(args.results_dir + '/modelS1_best.pth'))
        else:
            # createCAE for the S1 modality and pretrain it
            print("\nCreating and pretraining CAE for S1 modality...")
            modelS1 = Singlemodal_CAE(input_dim= args.n_channels2, output_dim=10, n_images= args.n_images2, mamba= args.mamba).to(device)
            optimizerS1 = torch.optim.Adam(modelS1.parameters(), lr=1e-4, weight_decay=args.weight_decay)
            loss_fn = nn.MSELoss().to(device)
            scalerS1 = torch.cuda.amp.GradScaler()

            best_loss = 9999.0
            if args.resume_pretrainS1 != '':
                print("=> loading checkpoint of S1 CAE '{}'".format(args.resume_pretrainS1))
                if os.path.isfile(args.resume_pretrainS1):
                    if args.gpu is None:
                        checkpoint = torch.load(args.resume_pretrainS1)
                    else:
                        loc = "cuda:{}".format(args.gpu)
                        checkpoint = torch.load(args.resume_pretrainS1, map_location=loc)
                    args.start_epoch = checkpoint['epoch']
                    modelS1.load_state_dict(checkpoint['state_dict'])
                    optimizerS1.load_state_dict(checkpoint['optimizer'])
                    best_loss = checkpoint['loss']
                    print(
                        "=> loaded checkpoint '{}' (epoch {})".format(
                            args.resume, checkpoint["epoch"]
                        )
                    )
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume_pretrainS1))

            datasetS1 = moco.loader.Singlemodal_Loader(
                listIDs = train_data_IDS,
                root = traindir,
                transform = transforms.Compose(augmentation),
                patch_size = args.patch_size,
                n_images = args.n_images2,
                n_channels = args.n_channels2,
                data_type = 'S1'
            )
            dataloaderS1 = torch.utils.data.DataLoader(
                datasetS1,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True,
            )
            resultsS1 = {'lr': [], 'train_loss': []}
            # pretrain CAE for modality 2 (S1)
            for epoch in range(1, args.epochs + 1):
                modelS1.train()
                total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloaderS1)
                for im in train_bar:
                    im = im.to(non_blocking=True, device=device)
                    # forward
                    output = modelS1(im)
                    loss = loss_fn(output, im)
                    # backward
                    optimizerS1.zero_grad()
                    scalerS1.scale(loss).backward()
                    scalerS1.step(optimizerS1)
                    scalerS1.update()
                    total_num += dataloaderS1.batch_size
                    total_loss += loss.item() * dataloaderS1.batch_size
                    train_bar.set_description(
                        'Pretrain S1 CAE Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, args.epochs,
                                                                            loss.item()))
                print("Pretrain S1 CAE - epoch total_loss/total_num", total_loss / total_num)
                resultsS1['lr'].append(optimizerS1.param_groups[0]['lr'])
                resultsS1['train_loss'].append(total_loss / total_num)
                # save statistics
                data_frame = pd.DataFrame(data=resultsS1, index=range(1, epoch + 1))
                data_frame.to_csv(args.results_dir + '/log_pretrainS1.csv', index_label='epoch')
                # save model
                torch.save({'epoch': epoch, 'state_dict': modelS1.state_dict(), 'optimizer': optimizerS1.state_dict(), 'loss': total_loss / total_num},
                        args.results_dir + '/modelS1_last.pth')
                # save best model
                if total_loss / total_num < best_loss:
                    shutil.copyfile(args.results_dir + '/modelS1_last.pth', args.results_dir + '/modelS1_best.pth')
                    best_loss = total_loss / total_num
            del modelS1
        torch.cuda.empty_cache()
        
        if not args.use_pretrained_S2:
            modelS2 = Singlemodal_CAE(input_dim= args.n_channels1, output_dim=10, n_images= args.n_images1, mamba= args.mamba).to(device)
            #load best S2 model
            if os.path.isfile(args.results_dir + '/modelS2_best.pth'):
                print("=> loading best checkpoint of S2 CAE '{}'".format(args.results_dir + '/modelS2_best.pth'))
                if args.gpu is None:
                    checkpoint = torch.load(args.results_dir + '/modelS2_best.pth')
                else:
                    loc = "cuda:{}".format(args.gpu)
                    checkpoint = torch.load(args.results_dir + '/modelS2_best.pth', map_location=loc)
                modelS2.load_state_dict(checkpoint['state_dict'])
                print("=> loaded best checkpoint of S2 CAE '{}'".format(args.results_dir + '/modelS2_best.pth'))
        if not args.use_pretrained_S1:
            modelS1 = Singlemodal_CAE(input_dim= args.n_channels2, output_dim=10, n_images= args.n_images2,mamba=args.mamba).to(device)
            #load best S1 model
            if os.path.isfile(args.results_dir + '/modelS1_best.pth'):
                print("=> loading best checkpoint of S1 CAE '{}'".format(args.results_dir + '/modelS1_best.pth'))
                if args.gpu is None:
                    checkpoint = torch.load(args.results_dir + '/modelS1_best.pth')
                else:
                    loc = "cuda:{}".format(args.gpu)
                    checkpoint = torch.load(args.results_dir + '/modelS1_best.pth', map_location=loc)
                modelS1.load_state_dict(checkpoint['state_dict'])
                print("=> loaded best checkpoint of S1 CAE '{}'".format(args.results_dir + '/modelS1_best.pth'))
        # create model MoCo model
        use_amp = True
        model = moco.builder.MoCo2encoders(
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

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        best_loss = 9999.0
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                if args.gpu is None:
                    checkpoint = torch.load(args.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(args.gpu)
                    checkpoint = torch.load(args.resume, map_location=loc)
                args.start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                best_loss = checkpoint['loss']
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True

        # logging
        results = {'lr': [], 'train_loss': []}

        train_dataset = moco.loader.MoCo2encodersLoader(
            listIDs = train_data_IDS,
            root = traindir,
            transform = transforms.Compose(augmentation),
            patch_size = args.patch_size,
            n_images1 = args.n_images1,
            n_channels1 = args.n_channels1,
            n_images2 = args.n_images2,
            n_channels2 = args.n_channels2
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

        validation_transform = transforms.Compose([
            # TO-IMPROVE: perdita di informazione in conversioni float64 -> float32 ??
            transforms.ToTensor(),
        ])

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        """#### MAIN LOOP """
        for epoch in range(args.start_epoch, args.epochs + 1):
            # train for one epoch
            train_loss = train(model, train_loader, optimizer, epoch, args, scaler, use_amp)
            print("MAIN LOOP - epoch # ", epoch, " train_loss", train_loss)
            results['train_loss'].append(train_loss)
            results['lr'].append(optimizer.param_groups[0]['lr'])

            """
            # TODO: REWRITE for segmentation testing over validation dataset.
            # up to now test_acc_1 is related to a classification task over CIFAR tested using a knn monitor
            test_acc_1 = test(model.autoencoder_q, memory_loader, validation_loader, epoch, args)
            results['test_acc@1'].append(test_acc_1)
            """

            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(args.start_epoch, epoch + 1))
            data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
            # save model
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },
                    args.results_dir + '/model_last.pth')
            # save best model
            if train_loss < best_loss:
                shutil.copyfile(args.results_dir + '/model_last.pth', args.results_dir + '/model_best.pth')
                best_loss = train_loss

            """
            print('Memory Usage at end epoch:', epoch)
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
            """

        # plot results
        #utils.results_plot(args.results_dir, 'log.csv')

        # for epoch in range(args.start_epoch, args.epochs):
        #     if args.distributed:
        #         train_sampler.set_epoch(epoch)
        #     adjust_learning_rate(optimizer, epoch, args)

        #     # train for one epoch
        #     train(train_loader, model, criterion, optimizer, epoch, args)

        #     if not args.multiprocessing_distributed or (
        #         args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        #     ):
        #         save_checkpoint(
        #             {
        #                 "epoch": epoch + 1,
        #                 "arch": args.arch,
        #                 "state_dict": model.state_dict(),
        #                 "optimizer": optimizer.state_dict(),
        #             },
        #             is_best=False,
        #             filename="checkpoint_{:04d}.pth.tar".format(epoch),
        #         )


    # def train(train_loader, model, criterion, optimizer, epoch, args) -> None:
    #     batch_time = AverageMeter("Time", ":6.3f")
    #     data_time = AverageMeter("Data", ":6.3f")
    #     losses = AverageMeter("Loss", ":.4e")
    #     top1 = AverageMeter("Acc@1", ":6.2f")
    #     top5 = AverageMeter("Acc@5", ":6.2f")
    #     progress = ProgressMeter(
    #         len(train_loader),
    #         [batch_time, data_time, losses, top1, top5],
    #         prefix="Epoch: [{}]".format(epoch),
    #     )

    #     # switch to train mode
    #     model.train()

    #     end = time.time()
    #     for i, (images, _) in enumerate(train_loader):
    #         # measure data loading time
    #         data_time.update(time.time() - end)

    #         if args.mmsingle:
    #             if args.gpu is not None:
    #                 images[0] = images[0].cuda(args.gpu, non_blocking=True)

    #         # compute output
    #         output, target = model(im_q=torch.reshape(images[0,:20,:,:],(1,2,10,256,256)), im_k=images[0,20:40,:,:])
    #         loss = criterion(output, target)

    #         # acc1/acc5 are (K+1)-way contrast classifier accuracy
    #         # measure accuracy and record loss
    #         acc1, acc5 = accuracy(output, target, topk=(1, 5))
    #         losses.update(loss.item(), images[0].size(0))
    #         top1.update(acc1[0], images[0].size(0))
    #         top5.update(acc5[0], images[0].size(0))

    #         # compute gradient and do SGD step
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()

    #         if i % args.print_freq == 0:
    #             progress.display(i)

    def train(net, data_loader, train_optimizer, epoch, args, scaler, use_amp=True):
        # switch to train mode
        net.train()
        adjust_learning_rate(train_optimizer, epoch, args)

        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

        for im_1, im_2 in train_bar:
            #im_1, im_2 = im_1.to(non_blocking=True, device=device), im_2.to(non_blocking=True, device=device)
            # im_1['im1'], im_1['im2'] = im_1['im1'].to(non_blocking=True, device=device), im_1['im2'].to(non_blocking=True, device=device)
            # im_2['im1'], im_2['im2'] = im_2['im1'].to(non_blocking=True, device=device), im_2['im2'].to(non_blocking=True, device=device)
            im_1 = im_1.to(non_blocking=True, device=device)
            im_2 = im_2.to(non_blocking=True, device=device)
            #print("im_1.shape", im_1.shape)  # torch.Size([32, 1, 1380, 128])
            #print('images dtype after im_1.to(non_blocking=True, device=device) in train() = {}'.format(im_1.dtype))

            """ 
            “Come pretext task di segmentazione potresti utilizzare la ricostruzione del radargramma. 
            Ovvero la rete deve essere in grado di riproporre in output il radargramma in ingresso”
            Parametri: Input= radargramma, Label = Lo stesso radargramma (aumentato)
            """
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = net(im_1, im_2)  ## CUDA out of memory: RuntimeError ERROR POINT
                print("TRAIN() function - minibatch loss", loss)
            del im_1, im_2

            # compute gradient and do SGD step
            train_optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(train_optimizer)
            scaler.update()
            # train_optimizer.step()

            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size
            print("train() function - minibatch cumulated total_loss", total_loss)

            train_bar.set_description(
                'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs,
                                                                        train_optimizer.param_groups[0]['lr'],
                                                                        total_loss / total_num))
            # gc.collect()
            # torch.cuda.empty_cache()

        print("train() function - epoch total_loss/total_num", total_loss / total_num)

        return total_loss / total_num

    def save_checkpoint(state, is_best, filename: str = "checkpoint.pth.tar") -> None:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, "model_best.pth.tar")


    class AverageMeter:
        """Computes and stores the average and current value"""

        def __init__(self, name, fmt: str = ":f") -> None:
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self) -> None:
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n: int = 1) -> None:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self) -> str:
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
            return fmtstr.format(**self.__dict__)


    class ProgressMeter:
        def __init__(self, num_batches, meters, prefix: str = "") -> None:
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix

        def display(self, batch) -> None:
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            print("\t".join(entries))

        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = "{:" + str(num_digits) + "d}"
            return "[" + fmt + "/" + fmt.format(num_batches) + "]"


    def adjust_learning_rate(optimizer, epoch, args) -> None:
        """Decay the learning rate based on schedule"""
        lr = args.lr
        if args.cos:  # cosine lr schedule
            lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.0
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    if __name__ == "__main__":
        main()
