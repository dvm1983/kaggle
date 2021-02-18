import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
from torchvision import models


class EffNetMSD(nn.Module):
    def __init__(self, net='efficientnet-b7', num_classes=5, advprop=False, n_dropouts=10, p=0.1):
        super().__init__()
        if net.split('-')[0] == 'efficientnet':
            self.model = EfficientNet.from_pretrained(net, advprop=advprop)
        elif net[0:7] == 'resnext':
            self.model = torch.hub.load('pytorch/vision:v0.6.0', net, pretrained=True)
        self.n_dropouts = n_dropouts
        self.dropout = nn.Dropout(p=p)
        self.linear = nn.Linear(1000, num_classes)
    def forward(self, x):
        model_output = self.model(x)
        out = self.dropout(self.linear(model_output))
        for i in range(self.n_dropouts - 1):
            out += self.dropout(self.linear(model_output))
        out /= self.n_dropouts
        return out