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
from metric import dice_coef, mIoU


labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.cuda()
    model.eval()

    iou = []
    dices = []
    with torch.no_grad():
        n_class = 21 #len(labels)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.to('cuda',dtype=torch.int64)
            masks = F.one_hot(masks.squeeze(1),num_classes=n_class).permute(0,3,1,2).to(torch.float32)[:,:,:,:]

            outputs = model(images)

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            criterion1, criterion2 = criterion
            bce_loss = criterion1(outputs, masks)
            focal_loss = criterion2(outputs, masks)
            loss = 0.5 * bce_loss + 0.5 * focal_loss
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            iou_item = mIoU(outputs, masks)
            iou.append(iou_item)
            dice = dice_coef(outputs, masks)
            dices.append(dice)
    
    miou = torch.cat(iou, 0)
    miou_per_class = torch.mean(miou, 0)
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(labels, miou_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    miou = torch.mean(miou_per_class).item()
    dices = torch.mean(dices_per_class).item()

    return miou, dices