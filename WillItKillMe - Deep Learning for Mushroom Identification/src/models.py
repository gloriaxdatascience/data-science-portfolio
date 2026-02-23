# src/models.py
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------
# Baseline CNN
# ---------------------------------------------------------------------
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128*16*16, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------
# ResNet18 Transfer Learning Model
# ---------------------------------------------------------------------
def load_resnet18(num_classes=2, dropout1=0.4, dropout2=0.3):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # freeze backbone
    for p in resnet.parameters():
        p.requires_grad = False

    # replace head
    resnet.fc = nn.Sequential(
        nn.Dropout(dropout1),
        nn.Linear(resnet.fc.in_features, 256), nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(256, num_classes)
    )

    return resnet
