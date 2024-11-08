import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os


img_transforms = A.Compose([
    A.Resize(height=512, width=512, interpolation=1, always_apply=True),
    A.Normalize(),
    ToTensorV2()
])

mask_transforms = A.Compose([
    A.Resize(height=512, width=512, interpolation=0, always_apply=True),
    ToTensorV2()
])

class PascalDataset(Dataset):
    def __init__(self, image_set='train', img_transforms=img_transforms, mask_transforms=mask_transforms, num=None):
        self.image_set = image_set
        self.img_transforms = img_transforms
        self.mask_transforms=mask_transforms
        if image_set == 'train':
            self.datalist = VOCSegmentation(root='.dataset/pascal_2012/',year=  "2012",image_set="train",download=True)
        elif image_set == 'validation':
            self.datalist = VOCSegmentation(root='.dataset/pascal_2012/',year=  "2012",image_set="val",download=True)

        if num is not None:
            self.datalist = self.datalist[:num]
        # if num is not None:
        #     self.images = sorted(os.listdir(datalist['images']))[:num]
        #     self.labels = sorted(os.listdir(datalist['targets']))[:num]
        # else:
        #     self.images = sorted(os.listdir(datalist['images']))
        #     self.labels = sorted(os.listdir(datalist['targets']))
        self.num_classes = 20+1  # pascal 클래스 수

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        img = np.array(self.datalist[index][0],dtype =np.float32)
        mask = np.array(self.datalist[index][1],dtype =np.int64)

        # img_path = os.path.join(self.images[index])
        # label_path = os.path.join(self.labels[index])

        # img = Image.open(img_path).convert("RGB")
        # mask = Image.open(label_path)
        
        # img = np.array(img)
        # mask = np.array(mask)

        # img = cv2.imread(img_path)
        # mask = cv2.imread(label_path)[:,:,0]
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # multi_mask= mask
        # 클래스별 마스크 생성
        # multi_mask = np.zeros((mask.shape[0], mask.shape[1], self.num_classes), dtype=np.float32)
        # multi_mask = np.zeros((mask.shape[0], mask.shape[1], self.num_classes))

        # for c in range(1, self.num_classes+1):
        #     multi_mask[:, :, c - 1] = (mask == c)
        
        # multi_mask = np.eye(self.num_classes)[mask.squeeze()]

        # if self.img_transforms:
        #     img = img_transforms(image=img)['image']
        #     # img = torch.from_numpy(img).permute(2,0,1).float()

        # if self.mask_transforms:
        #     multi_mask= mask_transforms(image=multi_mask)['image']
        #     multi_mask = multi_mask.to(dtype=torch.float32)
        #     # multi_mask = torch.from_numpy(multi_mask).permute(2,0,1).float()
        if self.img_transforms:
            img_mask = img_transforms(image = img, mask = mask)
            img = img_mask['image']
            multi_mask = img_mask['mask']

        return img, multi_mask
    
def get_pascal_dataloader(root = 'Pascal', train_batch_size=4, valid_batch_size = 2 ,num = None):
    if root == 'Pascal':
        train_dataset = PascalDataset(image_set='train', img_transforms=img_transforms, mask_transforms=mask_transforms,num=num)
        valid_dataset = PascalDataset(image_set='validation', img_transforms=img_transforms, mask_transforms=mask_transforms,num=num)
        
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