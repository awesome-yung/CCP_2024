import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
from tqdm.notebook import tqdm

# Seed 설정
import random
import numpy as np
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed = 42
seed_everything(seed)

# Data Augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    # transforms.GaussianNoise(mean=0.0, sigma=0.1, clip=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Dataset & Dataloader
train_data = ImageFolder('./tiny-imagenet-200/train', transform=transform_train)
val_data = ImageFolder('./tiny-imagenet-200/val', transform=transform)

batch_size = 256
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

# Model
device = 'cuda'
model = torchvision.models.resnet18(pretrained=False, num_classes=200)
model.fc = nn.Sequential(
    nn.Dropout(p=0.1),  # Dropout 추가
    nn.Linear(model.fc.in_features, 200)
)
model.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # weight_decay 추가
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

# Training Loop
epochs = 100
import wandb
wandb.init(entity='2024CCP', project='yumin', name='ImageNet_resnet18')

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        wandb.log({'Train Loss': round(loss.item(), 4)})

    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

    scheduler.step()

    wandb.log({
        'Train Acc': round(epoch_accuracy.item(), 4),
        'Val Acc': round(epoch_val_accuracy.item(), 4),
        'Learning Rate': optimizer.param_groups[0]['lr']
    })

    print(
        f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_accuracy:.4f} - "
        f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_accuracy:.4f}"
    )
