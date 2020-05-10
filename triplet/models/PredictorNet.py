import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as torchinit
import math
from torch.nn import init, Parameter
import copy
import pdb
import utils

from models import base

class SpatioTemporalNet(nn.Module):

    def __init__(self, num_frames, num_out_frames):
        super(SpatioTemporalNet, self).__init__()
        self.num_in_frames = 3
        self.num_out_frames = num_out_frames
        self.num_frames = num_frames
        self.num_layers = 4
        self.strides = [1,1,1,1]
        self.feature_maps_size = [64,128,256,512]
        self.relu = nn.ReLU(inplace=True)

        # CityScapes Image Encoder
        self.conv1_2D = base.conv3x3(self.num_in_frames, self.feature_maps_size[0], self.strides[0])
        self.bn1_2D = nn.BatchNorm2d(self.feature_maps_size[0])
        self.conv1_3D = base.conv1x1(self.feature_maps_size[0], self.feature_maps_size[0], self.strides[0])
        self.bn1_3D = nn.BatchNorm2d(self.feature_maps_size[0])

        self.conv2_2D = base.conv3x3(self.feature_maps_size[0], self.feature_maps_size[1], self.strides[0])
        self.bn2_2D = nn.BatchNorm2d(self.feature_maps_size[1])
        self.conv2_3D = base.conv1x1(self.feature_maps_size[1], self.feature_maps_size[1], self.strides[0])
        self.bn2_3D = nn.BatchNorm2d(self.feature_maps_size[1])

        self.conv3_2D = base.conv3x3(self.feature_maps_size[1], self.feature_maps_size[2], self.strides[2])
        self.bn3_2D = nn.BatchNorm2d(self.feature_maps_size[2])
        self.conv3_3D = base.conv1x1(self.feature_maps_size[2], self.feature_maps_size[2], self.strides[2])
        self.bn3_3D = nn.BatchNorm2d(self.feature_maps_size[2])

        self.conv4_2D = base.conv3x3(self.feature_maps_size[2], self.feature_maps_size[3], self.strides[3])
        self.bn4_2D = nn.BatchNorm2d(self.feature_maps_size[3])
        self.conv4_3D = base.conv1x1(self.feature_maps_size[3], self.feature_maps_size[3], self.strides[3])
        self.bn4_3D = nn.BatchNorm2d(self.feature_maps_size[3])

        # Cityscapes Image Encoder #2
        self.sat_conv1_2D = base.conv3x3(self.num_in_frames, self.feature_maps_size[0], self.strides[0])
        self.sat_bn1_2D = nn.BatchNorm2d(self.feature_maps_size[0])
        self.sat_conv1_3D = base.conv1x1(self.feature_maps_size[0], self.feature_maps_size[0], self.strides[0])
        self.sat_bn1_3D = nn.BatchNorm2d(self.feature_maps_size[0])

        self.sat_conv2_2D = base.conv3x3(self.feature_maps_size[0], self.feature_maps_size[1], self.strides[0])
        self.sat_bn2_2D = nn.BatchNorm2d(self.feature_maps_size[1])
        self.sat_conv2_3D = base.conv1x1(self.feature_maps_size[1], self.feature_maps_size[1], self.strides[0])
        self.sat_bn2_3D = nn.BatchNorm2d(self.feature_maps_size[1])

        self.sat_conv3_2D = base.conv3x3(self.feature_maps_size[1], self.feature_maps_size[2], self.strides[2])
        self.sat_bn3_2D = nn.BatchNorm2d(self.feature_maps_size[2])
        self.sat_conv3_3D = base.conv1x1(self.feature_maps_size[2], self.feature_maps_size[2], self.strides[2])
        self.sat_bn3_3D = nn.BatchNorm2d(self.feature_maps_size[2])

        self.sat_conv4_2D = base.conv3x3(self.feature_maps_size[2], self.feature_maps_size[3], self.strides[3])
        self.sat_bn4_2D = nn.BatchNorm2d(self.feature_maps_size[3])
        self.sat_conv4_3D = base.conv1x1(self.feature_maps_size[3], self.feature_maps_size[3], self.strides[3])
        self.sat_bn4_3D = nn.BatchNorm2d(self.feature_maps_size[3])

        # 3D Convolutional Layers
        self.conv3D_T = base.conv1x1_3D(self.feature_maps_size[3], self.feature_maps_size[3], self.strides[3])
        self.bn3D_T = nn.BatchNorm2d(self.feature_maps_size[3])
        self.conv1D_T = base.conv1x1(self.feature_maps_size[3], self.feature_maps_size[3], self.strides[3])
        self.bn1D_T = nn.BatchNorm2d(self.feature_maps_size[3])

        # Decoder Layers
        self.dconv1_2D = base.deconv3x3(self.feature_maps_size[3], self.feature_maps_size[2], self.strides[0])
        self.bn5_2D = nn.BatchNorm2d(self.feature_maps_size[2])
        self.econv1_2D = base.conv3x3(self.feature_maps_size[2], self.feature_maps_size[2], self.strides[0])
        self.ebn5_2D = nn.BatchNorm2d(self.feature_maps_size[2])

        self.dconv2_2D = base.deconv3x3(self.feature_maps_size[3], self.feature_maps_size[1], self.strides[0])
        self.bn6_2D = nn.BatchNorm2d(self.feature_maps_size[1])
        self.econv2_2D = base.conv3x3(self.feature_maps_size[1], self.feature_maps_size[1], self.strides[0])
        self.ebn6_2D = nn.BatchNorm2d(self.feature_maps_size[1])

        self.dconv3_2D = base.deconv3x3(self.feature_maps_size[2], self.feature_maps_size[0], self.strides[0])
        self.bn7_2D = nn.BatchNorm2d(self.feature_maps_size[0])
        self.econv3_2D = base.conv3x3(self.feature_maps_size[0], self.feature_maps_size[0], self.strides[0])
        self.ebn7_2D = nn.BatchNorm2d(self.feature_maps_size[0])

        self.dconv4_2D = base.deconv3x3(self.feature_maps_size[1], self.num_out_frames, self.strides[0])

        self.pool = nn.AvgPool2d(2, padding=0)


    def forward(self, x):
        enc_x4_all = []
        # CityScapes Frames
        for frame_idx in range(1, x.shape[1], 1):
            enc_x1 = self.relu(self.bn1_2D(self.conv1_2D(x[:, frame_idx, :, :, :])))
            enc_x1 = self.pool(self.relu(self.bn1_3D(self.conv1_3D(enc_x1))))

            enc_x2 = self.relu(self.bn2_2D(self.conv2_2D(enc_x1)))
            enc_x2 = self.pool(self.relu(self.bn2_3D(self.conv2_3D(enc_x2))))

            enc_x3 = self.relu(self.bn3_2D(self.conv3_2D(enc_x2)))
            enc_x3 = self.pool(self.relu(self.bn3_3D(self.conv3_3D(enc_x3))))

            enc_x4 = self.relu(self.bn4_2D(self.conv4_2D(enc_x3)))
            enc_x4_all.append(self.pool(self.relu(self.bn4_3D(self.conv4_3D(enc_x4)))))

        # Temporal Module
        enc_x4_all = torch.stack(enc_x4_all, dim=2)
        enc_x4_all = self.conv3D_T(enc_x4_all)
        enc_x4_all = self.relu(self.bn3D_T(enc_x4_all.reshape((enc_x4_all.shape[0],
                    enc_x4_all.shape[1]*enc_x4_all.shape[2], enc_x4_all.shape[3], enc_x4_all.shape[4]))))
        enc_x4_all = self.relu(self.bn1D_T(self.conv1D_T(enc_x4_all)))

        # Satellite Image
        sat_enc_x1 = self.relu(self.sat_bn1_2D(self.sat_conv1_2D(x[:, 0, :, :, :])))
        sat_enc_x1 = self.pool(self.relu(self.sat_bn1_3D(self.sat_conv1_3D(sat_enc_x1))))

        sat_enc_x2 = self.relu(self.sat_bn2_2D(self.sat_conv2_2D(sat_enc_x1)))
        sat_enc_x2 = self.pool(self.relu(self.sat_bn2_3D(self.sat_conv2_3D(sat_enc_x2))))

        sat_enc_x3 = self.relu(self.sat_bn3_2D(self.sat_conv3_2D(sat_enc_x2)))
        sat_enc_x3 = self.pool(self.relu(self.sat_bn3_3D(self.sat_conv3_3D(sat_enc_x3))))

        sat_enc_x4 = self.relu(self.sat_bn4_2D(self.sat_conv4_2D(sat_enc_x3)))
        weights = F.adaptive_avg_pool2d(sat_enc_x4, (1, 1))

        # Fusion
        enc_x4_all = enc_x4_all*weights

        # Decoder
        dec_x3 = self.relu(self.bn5_2D(self.dconv1_2D(enc_x4_all)))
        edec_x4 = self.relu(self.ebn5_2D(self.econv1_2D(enc_x3)))
        dec_x3 = torch.cat([dec_x3, edec_x4], dim=1)

        dec_x2 = self.relu(self.bn6_2D(self.dconv2_2D(dec_x3)))
        edec_x3 = self.relu(self.ebn6_2D(self.econv2_2D(enc_x2)))
        dec_x2 = torch.cat([dec_x2, edec_x3], dim=1)

        dec_x1 = self.relu(self.bn7_2D(self.dconv3_2D(dec_x2)))
        edec_x2 = self.relu(self.ebn7_2D(self.econv3_2D(enc_x1)))
        dec_x1 = torch.cat([dec_x1, edec_x2], dim=1)

        output = (self.dconv4_2D(dec_x1))

        return output
