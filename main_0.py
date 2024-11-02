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

import time
from BEIT import BEiT3
from dataset import get_dataloader
from model import get_model
from train import train_0, train_1
import wandb
from loss import FocalLoss

'''
11/2
dataset을 pascal 2012로 수정

main_0.py : root 주소 수정, learning rate, batch. 수정
dataset.py : VOC2012 클래스 추가. 라벨 다는 부분 임시 추가
train_0 : F.one_hot 부분 sqeeze 추가
valid.py : loss 부분 focal, BCE 계산되게 수정
valid_with_vis : 수정
model.py : num_classes = 21로 수정
'''

wandb.init(entity='2024CCP', project='yumin', name='float32')

RANDOM_SEED = 42

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False

set_seed()

LR = 1e-3
NUM_EPOCHS = 4
VAL_EVERY = 2
file_name = 'float32.pt'
train_batch_size = 8
valid_batch_size = 2

train_loader, valid_loader = get_dataloader(root ='./VOCdevkit/VOC2012/', train_batch_size=train_batch_size, valid_batch_size=valid_batch_size)

model = get_model()

criterion = [nn.BCELoss(),FocalLoss()]
optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-6)

train_0(model, train_loader, valid_loader, criterion, optimizer, NUM_EPOCHS, VAL_EVERY, file_name)

torch.cuda.empty_cache()