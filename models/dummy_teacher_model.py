
'''
Sourced from https://github.com/Matthew-Dickson/FilterBasedKnowledgeDistillation/tree/main
'''

import torch.nn as nn
from models.base_model import BaseModel
from models.layers.convolution import conv3x3


class DummyTeacherModel(BaseModel):
    def __init__(self,input_channels=1, num_classes=100):
        super().__init__()
        self.conv = conv3x3(input_channels, 32) 
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(32, 16)
        self.relu2 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(16)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out