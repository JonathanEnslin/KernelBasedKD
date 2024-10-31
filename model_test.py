from models.resnet import resnet20

import numpy as np

import torch
# Create random rank 3x64x64 tensor
# x = torch.randn(1, 3, 32, 32)
x = torch.randn(1, 3, 64, 64)
# x = torch.randn(1, 3, 128, 128)

# Create model
model = resnet20(num_classes=200, avgpooling_factor=2)
# model = resnet20(num_classes=200, conv1stride=2, conv1ksize=3, conv1padding=1)

# Forward pass
y = model(x)

# Print output shape
print(y.shape)
