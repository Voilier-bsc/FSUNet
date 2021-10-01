import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import PIL
from collections import OrderedDict
import cv2

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data


class ToLongTensor(object):
    def __call__(self, img):
        # img = torch.from_numpy(pic.transpose((2, 0, 1)))
        img = torch.from_numpy(img)
        return img.long()


class Resize(object):
    def __init__(self, nx, ny, mode=None):
        self.nx = nx
        self.ny = ny
        self.mode = mode

    def __call__(self, img):
        if self.mode == 'NEAREST':
            img = cv2.resize(img,(self.nx,self.ny),interpolation=cv2.INTER_NEAREST)
        else :
            img = cv2.resize(img,(self.nx,self.ny))
        return img


def LongTensorToNumpy(tensor, rgb_encoding, device):
    if len(tensor.size()) == 2:
        tensor.unsqueeze_(0)

    color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2)).to(device)

    for index, (class_name, color) in enumerate(rgb_encoding.items()):
        # Get a mask of elements equal to index
        mask = torch.eq(tensor[0], index)

        # Fill color_tensor with corresponding colors
        for channel, color_value in enumerate(color):
            color_tensor[channel].masked_fill_(mask, color_value)

    return color_tensor.to('cpu').detach().numpy().transpose(1,2,0)

def remap(image, old_values, new_values):

    tmp = np.zeros_like(image)
    for old, new in zip(old_values, new_values):

        if new != 0:
            tmp[image == old] = new

    return tmp