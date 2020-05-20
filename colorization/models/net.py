import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNet(nn.Module):
    def __init__(self, embedding_net, feature_size, img_len):
        super(TemporalNet, self).__init__()
        self.img_len = img_len
        #self.num_classes = math.factorial(img_len)
        # predict the central frame
        self.num_classes = img_len
        self.feature_size = feature_size

        self.feature_extractor = embedding_net
        self.classifier = nn.Linear(self.feature_size* self.img_len, self.num_classes)
        nn.init.kaiming_uniform_(self.classifier.weight)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        output1 = self.feature_extractor(x1)
        output2 = self.feature_extractor(x2)
        output3 = self.feature_extractor(x3)
        output1 = output1.view(output1.shape[0], -1)
        output2 = output1.view(output2.shape[0], -1)
        output3 = output1.view(output3.shape[0], -1)

        # dim = 1: see https://github.com/xudejing/video-clip-order-prediction
        features = torch.cat([output1, output2, output3], dim=1)
        x = self.classifier(features)
        #x = self.softmax(x)
        
        return x

    def get_embedding(self, x):
        return self.feature_extractor(x)


class ColorNet(nn.Module):
    def __init__(self, embedding_net, consecutiveFrame):
        super(ColorNet, self).__init__()
        self.embedding_net = embedding_net
        self.consecutiveFrame = consecutiveFrame
        self.relu = nn.ReLU()
        self.conv3d_1 = nn.Conv3d(2048, 128, kernel_size=(1, 2, 2), dilation=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(128, 128, kernel_size=(1, 2, 2), dilation=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(128, 128, kernel_size=(1, 2, 2), dilation=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(128, 128, kernel_size=(1, 1, 1), dilation=(1, 1, 1))
        self.conv3d_5 = nn.Conv3d(128, 128, kernel_size=(1, 1, 1), dilation=(1, 1, 1))

    def forward(self, x1, x2, x3, x4):
        out_1 = self.relu(self.embedding_net(x1))[:, :, None, :, :]
        out_2 = self.relu(self.embedding_net(x2))[:, :, None, :, :]
        out_3 = self.relu(self.embedding_net(x3))[:, :, None, :, :]
        out_4 = self.relu(self.embedding_net(x4))[:, :, None, :, :]
        out = torch.cat((out_1, out_2, out_3, out_4), axis=2)
        # out = out.reshape([-1, 4] + list(out.shape[1:]))
        # print(out.shape)
        # print(x1.shape)
        # print(self.embedding_net(x1).shape)
        # print(out.shape)
        out = self.relu(self.conv3d_1(out))
        out = self.relu(self.conv3d_2(out))
        out = self.relu(self.conv3d_3(out))
        out = self.relu(self.conv3d_4(out))
        out = self.relu(self.conv3d_5(out))
        out = out.transpose(2, 1)
        return out

        



