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
from valid import validation
import wandb


def save_model(model, file_name):
    SAVED_DIR = './saved/'
    if not os.path.exists(SAVED_DIR ):
        os.makedirs(SAVED_DIR )
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

def train_0(model, train_loader, valid_loader, criterion, optimizer, NUM_EPOCHS, VAL_EVERY, file_name):
    print(f'Start training..')
    model.cuda()

    n_class = 150# len(val_labels)
    best_miou = 0.

    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(train_loader):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()

            # inference
            outputs = model(images)

            # loss 계산
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                wandb.log({'Train Loss': round(loss.item(),4),
                           'epoch' : epoch+1,
                           'learning rate' : optimizer.param_groups[0]['lr']})

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            miou, dice = validation(epoch + 1, model, valid_loader, criterion, thr=0.5)
            wandb.log({'Validation miou': miou,
                       'Validation Dice': dice,
                       })

            # if best_dice < dice:
            if best_miou < miou:
                print(f"Best performance at epoch: {epoch + 1}, {best_miou:.4f} -> {miou:.4f}")
                print(f"Save model in {file_name}")
                best_miou = miou
                save_model(model, file_name)

def train_1(model, train_loader, valid_loader, criterion, optimizer, NUM_EPOCHS, VAL_EVERY, file_name):
    print(f'Start training..')
    model.cuda()

    n_class = 150# len(val_labels)
    best_miou = 0.

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(train_loader):

            images, masks = images.cuda(), masks.cuda()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                wandb.log({'Train Loss': round(loss.item(),4),
                           'epoch' : epoch+1,
                           'learning rate' : optimizer.param_groups[0]['lr']})

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            miou, dice = validation(epoch + 1, model, valid_loader, criterion, thr=0.5)
            wandb.log({'Validation miou': miou,
                       'Validation Dice': dice,
                       })

            # if best_dice < dice:
            if best_miou < miou:
                print(f"Best performance at epoch: {epoch + 1}, {best_miou:.4f} -> {miou:.4f}")
                print(f"Save model in {file_name}")
                best_miou = miou
                save_model(model, file_name)

def train_2(model, train_loader, valid_loader, criterion, optimizer, NUM_EPOCHS, VAL_EVERY, file_name):
    print(f'Start training..')
    model.cuda()

    n_class = 150# len(val_labels)
    best_miou = 0.

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(train_loader):

            images, masks = images.cuda(), masks.cuda()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                wandb.log({'Train Loss': round(loss.item(),4),
                           'epoch' : epoch+1,
                           'learning rate' : optimizer.param_groups[0]['lr']})

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            miou, dice = validation(epoch + 1, model, valid_loader, criterion, thr=0.5)
            wandb.log({'Validation miou': miou,
                       'Validation Dice': dice,
                       })

            # if best_dice < dice:
            if best_miou < miou:
                print(f"Best performance at epoch: {epoch + 1}, {best_miou:.4f} -> {miou:.4f}")
                print(f"Save model as {file_name}")
                best_miou = miou
                save_model(model, file_name)