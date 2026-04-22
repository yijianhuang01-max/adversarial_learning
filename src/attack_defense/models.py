from __future__ import annotations

import torch.nn as nn


def build_resnet18(num_classes: int = 10) -> nn.Module:
    from torchvision import models

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
