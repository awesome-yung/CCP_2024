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

img_transforms = A.Compose([
    A.Resize(height=224, width=224, interpolation=1, always_apply=True),
    ToTensorV2()
])

mask_transforms = A.Compose([
    A.Resize(height=224, width=224, interpolation=0, always_apply=True),
])

class ADE20KDataset(Dataset):
    def __init__(self, root, image_set='train', img_transforms=img_transforms, mask_transforms=mask_transforms):
        self.root = root
        self.image_set = image_set
        self.img_transforms = img_transforms
        self.mask_transforms=mask_transforms

        self.images_dir = os.path.join(root, 'images', image_set)
        self.labels_dir = os.path.join(root, 'annotations', image_set)

        # self.images = sorted(os.listdir(self.images_dir))[:500]
        # self.labels = sorted(os.listdir(self.labels_dir))[:500]

        self.images = sorted(os.listdir(self.images_dir))
        self.labels = sorted(os.listdir(self.labels_dir))
        self.num_classes = 150  # ADE20K 클래스 수

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.images[index])
        label_path = os.path.join(self.labels_dir, self.labels[index])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path)

        img = np.array(img).astype(np.float32)
        mask = np.array(mask)

        # 클래스별 마스크 생성
        multi_mask = np.zeros((mask.shape[0], mask.shape[1], self.num_classes), dtype=np.float32)

        for c in range(1, self.num_classes+1):
            multi_mask[:, :, c - 1] = (mask == c)

        if self.img_transforms:
            img = img_transforms(image=img)['image']

        if self.mask_transforms:
            multi_mask= mask_transforms(image=multi_mask)['image']
            multi_mask = torch.from_numpy(multi_mask).permute(2,0,1)

        return img, multi_mask
    
def get_dataloader(root = './ADE20K/ADEChallengeData2016', train_batch_size=4, valid_batch_size = 2 ):

    train_dataset = ADE20KDataset(root=root, image_set='training', img_transforms=img_transforms, mask_transforms=mask_transforms)
    valid_dataset = ADE20KDataset(root=root, image_set='validation', img_transforms=img_transforms, mask_transforms=mask_transforms)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    return train_loader, valid_loader