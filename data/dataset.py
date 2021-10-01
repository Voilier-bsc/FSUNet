# %% 

import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from collections import OrderedDict
import cv2
from data.data_util import *

class Dataset(torch.utils.data.Dataset):
    origin_classes = (255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255, 255, 255, 5, 255, 6,
                   7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 16, 17, 18, -1)

    new_classes = (0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7,
                   8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0)

    color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232)),
            ('building', (70, 70, 70)),
            ('wall', (102, 102, 156)),
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32))
    ])

    def __init__(self, data_dir, mode = 'train', transform=None, label_transform=None, depth_transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.depth_transform = depth_transform
        self.input_Image_list = open(os.path.join(data_dir,"{}Images.txt".format(self.mode))).read().split('\n')
        self.Depth_list = open(os.path.join(data_dir,"{}Depth.txt".format(self.mode))).read().split('\n')
        self.Label_list = open(os.path.join(data_dir,"{}Labels.txt".format(self.mode))).read().split('\n')

    
    def __getitem__(self, idx):
        img = plt.imread(os.path.join(self.data_dir, self.input_Image_list[idx]))
        
        label = plt.imread(os.path.join(self.data_dir, self.Label_list[idx])).squeeze()*255
        depth = plt.imread(os.path.join(self.data_dir, self.Depth_list[idx])).squeeze()

        label = remap(label, self.origin_classes, self.new_classes)

        if self.transform:
            img = self.transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.depth_transform:
            depth = self.depth_transform(depth)
        
        data = {'input' : img, 'label' : label, 'depth' : depth}

        return data


    def __len__(self):
        return len(self.input_Image_list)

