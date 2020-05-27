# code based on https://github.com/ColasGael/QA-squad

import torch

import torch.nn as nn
import torch.nn.functional as F


class RatioLoss(nn.Module):
    def __init__(self):
        super(RatioLoss, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.abs(input - target))

class CentroidLoss(nn.Module):
    def __init__(self):
        super(CentroidLoss, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.abs(input - target))