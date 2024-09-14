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
from model import get_model, get_model_with_flash
from train import train_0, train_1, train_2
import wandb

wandb.init(entity='2024CCP', project='yumin', name='flash')

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
NUM_EPOCHS = 3
VAL_EVERY = 1
SAVED_DIR = './'
train_batch_size = 4
valid_batch_size = 2

train_loader, valid_loader = get_dataloader(root ='./ADE20K/ADEChallengeData2016', train_batch_size=train_batch_size, valid_batch_size=valid_batch_size)

model = get_model_with_flash()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-6)

train_0(model, train_loader, valid_loader, criterion, optimizer, NUM_EPOCHS, VAL_EVERY, SAVED_DIR)
