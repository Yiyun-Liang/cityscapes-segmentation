import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNet(nn.Module):
    def __init__(self, embedding_net, feature_size, img_len):
        super(TemporalNet, self).__init__()
        self.img_len = img_len
        self.num_classes = math.factorial(img_len)
        self.pair_num = int(img_len*(img_len-1)/2)
        # predict the central frame
        #self.num_classes = img_len
        self.feature_size = feature_size

        self.feature_extractor = embedding_net
        self.pairwise = nn.Linear(self.feature_size * 2, 512)
        self.classifier = nn.Linear(512 * self.pair_num, self.num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.kaiming_uniform_(self.pairwise.weight)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        output1 = self.feature_extractor(x1)
        output2 = self.feature_extractor(x2)
        output3 = self.feature_extractor(x3)

        output1 = output1.view(output1.shape[0], -1)
        output2 = output2.view(output2.shape[0], -1)
        output3 = output3.view(output3.shape[0], -1)

        # dim = 1: see https://github.com/xudejing/video-clip-order-prediction
        pair12 = self.pairwise(torch.cat([output1, output2], dim=1))
        pair13 = self.pairwise(torch.cat([output1, output3], dim=1))
        pair23 = self.pairwise(torch.cat([output2, output3], dim=1))

        pair12 = self.relu(pair12)
        pair13 = self.relu(pair13)
        pair23 = self.relu(pair23)

        features = torch.cat([pair12, pair13, pair23], dim=1)
        x = self.dropout(features)
        x = self.classifier(x)
        #x = self.softmax(x)
        
        return x

    def get_embedding(self, x):
        return self.feature_extractor(x)
