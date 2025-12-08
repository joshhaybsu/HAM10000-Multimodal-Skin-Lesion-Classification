import torch
import torch.nn as nn
from torchvision import models

class BaselineCNN(nn.Module):
    """
    Baseline Image-Only Classifier
    Using ResNet-18 with a modified final layer for binary classification.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Replace final FC layer with binary classifier
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)  # output: raw logit

    def forward(self, images):
        """
        Forward pass for images only.
        Output is a raw logit for BCEWithLogitsLoss.
        """
        logits = self.model(images)
        return logits.squeeze(1)  # shape: (batch)
