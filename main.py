import os
import numpy as np
from numpy.core.fromnumeric import resize

import torch
import torch.nn as nn
from torch.nn.modules import batchnorm
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from torchvision import transforms


from model.FSUNet import FSUNet
from data.dataset import *
from util import *
from loss import inverse_huber_loss

from config import ConfigParameters


configs = ConfigParameters()

## define train hyperparameters
mode = configs.mode
train_continue = configs.train_continue

lr = configs.lr
batch_size = configs.batch_size
num_epoch = configs.num_epoch
weight_decay = configs.weight_decay
lr_decay_epochs = configs.lr_decay_epochs
lr_decay = configs.lr_decay

ny = configs.ny
nx = configs.nx

data_dir = configs.data_dir
ckpt_dir = configs.ckpt_dir
log_dir = configs.log_dir
result_dir = configs.result_dir

encoder_relu = configs.encoder_relu
decoder_relu = configs.decoder_relu


## create directory
result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


## train
if mode == 'train':
    transform_train_img = transforms.Compose([Resize(nx,ny), transforms.ToTensor()])
    transform_train_seg = transforms.Compose([Resize(nx,ny,'NEAREST'), ToLongTensor()])
    transform_train_depth = transforms.Compose([Resize(nx,ny), transforms.ToTensor()])

    transform_val_img = transforms.Compose([Resize(nx,ny), transforms.ToTensor()])
    transform_val_seg = transforms.Compose([Resize(nx,ny,'NEAREST'), ToLongTensor()])
    transform_val_depth = transforms.Compose([Resize(nx,ny), transforms.ToTensor()])

    dataset_train = Dataset(data_dir=data_dir, mode = 'train', transform=transform_train_img, label_transform=transform_train_seg, depth_transform=transform_train_depth)
    dataset_val = Dataset(data_dir=data_dir, mode = 'val', transform=transform_val_img, label_transform=transform_val_seg, depth_transform=transform_val_depth)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    # variable setting
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

## class encoding
class_encoding = dataset_train.color_encoding
num_classes = len(class_encoding)

## model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = FSUNet(num_classes=num_classes, encoder_relu=encoder_relu, decoder_relu=decoder_relu).to(device)

## loss
CrossEntropyLoss = nn.CrossEntropyLoss().to(device)

## optimizer
optim = torch.optim.Adam(network.parameters(), lr=lr, weight_decay = weight_decay)

## lr scheduler
lr_updater = lr_scheduler.StepLR(optim, lr_decay_epochs, lr_decay)

## extra function
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)[0]
fn_denorm = lambda x, mean, std: (x * std) + mean

## Tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## train
st_epoch = 0

if mode == 'train':
    if train_continue == 'on':
        network, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=network, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        network.train()
        loss_arr = []

        for batch, data in enumerate(loader_train):
            # forward pass
            input = data['input'].to(device)
            label = data['label'].to(device)
            depth = data['depth'].to(device)

            
            out_seg, out_depth = network(input)
        
            # backward pass
            optim.zero_grad()

            seg_loss = CrossEntropyLoss(out_seg, label)
            depth_loss = inverse_huber_loss(out_depth, depth)

            loss = seg_loss + depth_loss
            loss.backward()

            optim.step()
            lr_updater.step()

            loss_arr += [loss.item()]



            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

                
            if batch % 10 == 0:
                # use Tensorboard          
                input = fn_tonumpy(input)
                label = LongTensorToNumpy(label, class_encoding, device)
                depth = fn_tonumpy(depth)
                _, out_seg = torch.max(out_seg.data, 1)
                out_seg = LongTensorToNumpy(out_seg, class_encoding, device)
                out_depth = fn_tonumpy(out_depth)

                id = num_batch_train * (epoch - 1) + batch

                writer_train.add_image('input', input, id, dataformats='HWC')
                writer_train.add_image('label_seg', label, id, dataformats='HWC')
                writer_train.add_image('out_seg', out_seg, id, dataformats='HWC')
                writer_train.add_image('label_depth', depth, id, dataformats='HWC')
                writer_train.add_image('out_depth', out_depth, id, dataformats='HWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
            
        with torch.no_grad():
            network.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val):
                input = data['input'].to(device)
                label = data['label'].to(device)
                depth = data['depth'].to(device)

                out_seg, out_depth = network(input)

                seg_loss = CrossEntropyLoss(out_seg, label)
                depth_loss = inverse_huber_loss(out_depth, depth)

                loss = seg_loss + depth_loss
                
                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                if batch % 10 == 0:
                    # use Tensorboard          
                    input = fn_tonumpy(input)
                    label = LongTensorToNumpy(label, class_encoding, device)
                    depth = fn_tonumpy(depth)
                    _, out_seg = torch.max(out_seg.data, 1)
                    out_seg = LongTensorToNumpy(out_seg, class_encoding, device)
                    out_depth = fn_tonumpy(out_depth)

                    id = num_batch_val * (epoch - 1) + batch

                    writer_val.add_image('input', input, id, dataformats='HWC')
                    writer_val.add_image('label_seg', label, id, dataformats='HWC')
                    writer_val.add_image('out_seg', out_seg, id, dataformats='HWC')
                    writer_val.add_image('label_depth', depth, id, dataformats='HWC')
                    writer_val.add_image('out_depth', out_depth, id, dataformats='HWC')


        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 20 == 0:
            save(ckpt_dir=ckpt_dir, net=network, optim=optim, epoch=epoch)
    
    writer_train.close()
    writer_val.close()