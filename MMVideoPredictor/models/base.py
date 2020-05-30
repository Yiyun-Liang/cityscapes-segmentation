import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.utils as torchutils
from torch.nn import init, Parameter

import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 2D convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "3x3 2D convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv1x1_3D(in_planes, out_planes, stride=1):
    "1x1 3D convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1)) #, stride=stride, padding=(1,0,0), bias=False)

def deconv3x3(in_planes, out_planes, stride=1):
    "3x3 2D deconvolution"
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=(2,2), stride=2, padding=0)
