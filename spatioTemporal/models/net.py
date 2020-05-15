import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNet(nn.Module):
    def __init__(self, embedding_net, feature_size, img_len):
        super(TemporalNet, self).__init__()
        self.img_len = img_len
        self.num_classes = math.factorial(img_len)
        self.feature_size = feature_size

        self.feature_extractor = embedding_net
        self.classifier = nn.Linear(self.feature_size* self.img_len, self.num_classes)
        #self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

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
        #x = self.relu(x)
        #print(x.shape)
        x = self.softmax(x)
        
        return x

    def get_embedding(self, x):
        return self.feature_extractor(x)
