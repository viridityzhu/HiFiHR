import torch
import torch.nn as nn
import torch.nn.functional as F
from network.efficientnet_pt.model import EfficientNet

class EffiEncoder(nn.Module):
    def __init__(self, pretrain='effb3'):
        super(EffiEncoder, self).__init__()
        self.pretrain = pretrain
        if self.pretrain == 'effb3':
            self.encoder = EfficientNet.from_pretrained('efficientnet-b3')
            # b3 [1536,7,7]
            self.pool = nn.AvgPool2d(7, stride=1)

    def forward(self, x):
        features, low_features = self.encoder.extract_features(x)#[B,1536,7,7] = 75264,  [B,32,56,56] = 100352
        features = self.pool(features)
        features = features.view(features.shape[0],-1)##[B,1536]
        return low_features, features