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

import time
from BEIT import BEiT3
from dataset import get_dataloader
from model import get_model
from train import train_0, train_1, train_2
import wandb

wandb.init(entity='2024CCP', project='yumin', name='float16')

RANDOM_SEED = 42

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False

set_seed()

LR = 1e-5
NUM_EPOCHS = 1
VAL_EVERY = 1
file_name = 'float16.pt'
train_batch_size = 4
valid_batch_size = 1

train_loader, valid_loader = get_dataloader(root ='./ADE20K/ADEChallengeData2016', train_batch_size=train_batch_size, valid_batch_size=valid_batch_size)

model = get_model()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-6)

train_1(model, train_loader, valid_loader, criterion, optimizer, NUM_EPOCHS, VAL_EVERY, file_name)

torch.cuda.empty_cache()
