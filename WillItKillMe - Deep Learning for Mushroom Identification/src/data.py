import os
from pathlib import Path
from PIL import Image

from torchvision.datasets import ImageFolder
from torchvision import transforms


# ---------------------------------------------------------------------
# Safe Windows loader (handles long paths)
# ---------------------------------------------------------------------
def win_safe_loader(path: str):
    if os.name == 'nt' and not path.startswith('\\\\?\\'):
        path = '\\\\?\\' + path
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# ---------------------------------------------------------------------
# Safe ImageFolder wrapper
# ---------------------------------------------------------------------
class SafeImageFolder(ImageFolder):
    def __init__(self, root, transform=None, **kwargs):
        super().__init__(root=root, transform=transform,
                         loader=win_safe_loader, **kwargs)


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_baseline_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def get_resnet_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf

def get_transforms(model_type="resnet"):
    if model_type == "baseline":
        _, val_tf = get_baseline_transforms()
        return val_tf
    elif model_type == "resnet":
        _, val_tf = get_resnet_transforms()
        return val_tf
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

