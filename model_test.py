from models.resnet import resnet20

import numpy as np

import torch
print('wahat')
# Create random rank 3x64x64 tensor
# x = torch.randn(1, 3, 32, 32)
x = torch.randn(1, 3, 64, 64)
# x = torch.randn(1, 3, 128, 128)

# Create model
model = resnet20(num_classes=10, conv1stride=2, conv1ksize=5, conv1padding=2)

# Forward pass
y = model(x)

# Print output shape
print(y.shape)
