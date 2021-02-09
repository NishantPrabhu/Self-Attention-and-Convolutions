"""
Model classes - resnet, attention-cnn, SAN, VIT, 
Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, zero_init_residual=False):
        super(ResNet18, self).__init__()

        # Initial layers
        conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu1 = nn.ReLU()

        # Backbone
        resnet = models.resnet18(pretrained=False)
        layers = list(resnet.children())
        self.backbone = nn.Sequential(conv0, bn1, relu1, *layers[4 : len(layers) - 1])
        self.backbone_dim = 512

        # Weight initialization
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Reference: https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.backbone.modules():
                if isinstance(m, models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        return out

class ClassificationHead(nn.Module):
    def __init__(self, in_dim=512, n_classes=10):
        super(ClassificationHead, self).__init__()
        self.W1 = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        x = self.W1(x)
        return F.log_softmax(x, dim=-1)

