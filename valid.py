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



label_path = './ADE20K/ADEChallengeData2016/objectInfo150.txt'
labels = []
val_labels = []
with open(label_path, 'r') as file:
    for line in file:
        labels.append(line.split("\t")[-1].strip())
        val_labels.append(line.split("\t")[-1].strip())
        
labels[0] = 'background'
del val_labels[0]


def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.cuda()
    model.eval()

    iou = []
    dices = []
    with torch.no_grad():
        n_class = 20+1#len(labels)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # images, masks = images.cuda(), masks.to('cuda',dtype=torch.int64)
            # masks = F.one_hot(masks,num_classes=n_class).permute(0,3,1,2).to(torch.float32)[:,1:,:,:]
            images, masks = images.cuda(), masks.to('cuda',dtype=torch.int64)
            multi_mask = torch.zeros((masks.shape[0], n_class,masks.shape[1],masks.shape[2] )).to('cuda',torch.float32)

            for c in range(1, n_class):
                multi_mask[:, c - 1, :,:] = (masks == c)


            masks = multi_mask

            outputs = model(images)

            loss = criterion(outputs, masks)
            total_loss += loss

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")


            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            # outputs = torch.argmax(outputs,dim=1).detach().cpu()
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
        for c, d in zip(val_labels, miou_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    miou = torch.mean(miou_per_class).item()
    dices = torch.mean(dices_per_class).item()

    return miou, dices

if __name__ == '__main__':
    from dataset import get_dataloader
    criterion = nn.BCELoss()

    model_path = './saved/epoch_0_float32.pt'
    model = torch.load(model_path)
    _, valid_loader = get_dataloader(root ='./ADE20K/ADEChallengeData2016', train_batch_size=1, valid_batch_size=4)

    validation(0,model,valid_loader, criterion,0.5)