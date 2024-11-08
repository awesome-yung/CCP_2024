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
from model import get_model

import datetime
from tqdm import tqdm

import albumentations as A

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
        
        # (edited) visualize options
        output_path = './vis/'
        vis = True
        save_cnt = 4
        
        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.to('cuda',dtype=torch.int64)
            # masks = F.one_hot(masks,num_classes=n_class).permute(0,3,1,2).to(torch.float32)[:,1:,:,:]

            # outputs = model(images)

            # output_h, output_w = outputs.size(-2), outputs.size(-1)
            # mask_h, mask_w = masks.size(-2), masks.size(-1)

            # # restore original size
            # if output_h != mask_h or output_w != mask_w:
            #     outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            multi_mask = torch.zeros((masks.shape[0], n_class,masks.shape[1],masks.shape[2] )).to('cuda',torch.float32)

            for c in range(1, n_class):
                multi_mask[:, c - 1, :,:] = (masks == c)


            masks = multi_mask

            outputs = model(images)

            # loss = criterion(outputs, masks)
            # total_loss += loss
            cnt += 1

            outputs = outputs.detach().cpu()
            # outputs = (outputs > thr).detach().cpu()
            
            # outputs = torch.argmax(outputs,dim=1).detach().cpu()
            masks = masks.detach().cpu()

            iou_item = mIoU(outputs, masks)
            iou.append(iou_item)
            dice = dice_coef(outputs, masks)
            dices.append(dice)
            # (edited) visualize
            if vis:
                for sample_num in range(images.shape[0]):
                    save_path = output_path+f'batch{cnt}_sample{sample_num}'+'.png'
                    outputs = outputs.to('cuda')
                    thr_map = outputs>thr
                    output_map = thr_map * outputs

                    fig = plt.figure(figsize=(30,10))
                    ax1 = fig.add_subplot(131)
                    ax2 = fig.add_subplot(132)
                    ax3 = fig.add_subplot(133)
                    img = images[sample_num,:,:,:].to('cpu').permute(1,2,0).numpy().astype(np.int8)
                    ax1.imshow(img)
                    ax2.imshow(visualizer(img,torch.argmax(masks[sample_num,:,:,:],dim=0,keepdim=True).to('cpu').permute(1,2,0).numpy()))
                    ax3.imshow(visualizer(img,torch.argmax(output_map[sample_num,:,:,:],dim=0,keepdim=True).to('cpu').permute(1,2,0).numpy()))

                    plt.savefig(save_path)
                    plt.close()
                ## to stop visualizing
                # if cnt == save_cnt:
                #     vis = False
                    
    
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

# (edited) visualize functions
def color_map(N=150, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def visualizer(img,mask):
    cmap = color_map()[:, np.newaxis, :]
    new_im = np.dot(mask == 0, cmap[0])
    for i in range(1, cmap.shape[0]):
        new_im += np.dot(mask == i, cmap[i])
    new_im = Image.fromarray(new_im.astype(np.uint8))
    blend_image = Image.blend(Image.fromarray(img.astype(np.uint8)), new_im, alpha=0.8)
    return np.array(blend_image)


# image_path = './ADE20K/ADEChallengeData2016/images/validation/ADE_val_00000265.jpg'
# model_path = './saved/float32.pt'

# image = Image.open(image_path).convert('RGB')

# model = get_model()

# model.load_state_dict(torch.load(model_path))

# model.eval()

# image = Image.open(image_path).convert('RGB')
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # 필요한 경우 이미지 크기 조정
#     transforms.ToTensor(),
# ])
# image = transform(image).unsqueeze(0).cuda()

# with torch.no_grad():
#     output = model(image)
#     output = torch.sigmoid(output)  # 확률 값으로 변환
#     output = (output > 0.5).cpu()  # threshold 적용

#     # 시각화
#     fig = plt.figure(figsize=(10, 10))
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
#     ax1.imshow(image.cpu().squeeze().permute(1, 2, 0))
#     ax2.imshow(visualizer(image.cpu().squeeze(), output.squeeze()))
#     plt.show()

if __name__ =='__main__':
    from pascal_dataset import get_pascal_dataloader
    criterion = nn.BCELoss()

    # model_path = './saved/epoch_48_float32.pt'  
    model_path = './saved/epoch_50_float32.pt'
    model = torch.load(model_path)
    _, valid_loader = get_pascal_dataloader(root ='Pascal', train_batch_size=1, valid_batch_size=4)

    validation(0,model,valid_loader, criterion,0.1)