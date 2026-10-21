"""Submission Preview 2: EfficientNet-B0 dengan classifier head 5 kelas.

Arsitektur model menggunakan EfficientNet-B0 dari torchvision.
Output model: logits [batch, 5], urutan kelas (alfabetis):
  0: botol_plastik, 1: gelas_plastik, 2: kertas_kardus, 3: saset_kemasan, 4: styrofoam
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5


def get_model(num_classes=NUM_CLASSES):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
