import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim

import datetime
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2) # ( b, c, h, w ) -> ( b, c, h*w )
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1) # ( b, c, h*w ) -> ( b, c )

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps) # ( b, c )

def mIoU(y_true, y_pred):

    y_true_f = y_true.flatten(2).bool()  # ( b, h, w, c ) -> ( b, h*w, c )
    y_pred_f = y_pred.flatten(2).bool()
    intersection = torch.sum(y_true_f * y_pred_f, -1).float()
    union = torch.sum(y_true_f | y_pred_f, -1).float()

    eps = 1e-10
    iou = (intersection + eps) / (union + eps)

    return iou