'''
Sourced from https://github.com/Matthew-Dickson/FilterBasedKnowledgeDistillation/tree/main
'''

from models.base_model import BaseModel
from models.layers.convolution import conv3x3
import torch.nn as nn


class DummyStudentModel(BaseModel):
    def __init__(self,input_channels=1, num_classes=100):
        super().__init__()
        self.conv = conv3x3(input_channels, 32) 
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(30)
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out