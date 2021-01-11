
"""
Some custom loss functions.

Authors: Nishant Prabhu, Mukund Varma T
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F


class ClassificationLoss(nn.Module):

    def __init__(self, smoothing=None):
        super().__init__()
        self.eps = 0.0 if (smoothing is None) else smoothing

    def forward(self, output, target):
        w = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
        w = w * (1-self.eps) + (1-w) * self.eps / (output.size(1) - 1)
        log_prob = F.log_softmax(output, dim=1)
        loss = (-w * log_prob).sum(dim=1).mean()
        return loss