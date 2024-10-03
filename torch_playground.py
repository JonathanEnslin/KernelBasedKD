import torch
import torch.nn.functional as F


# Create a random 5x4x3x3 tensor
x = torch.rand(6, 3, 3, 3)
print(x)
print(x.sum(dim=0).shape)  # torch.Size([4, 3, 3])
print(x.sum(dim=1).shape)  # torch.Size([5, 3, 3])
print(x.sum(dim=(0, 1)).shape)  # torch.Size([5, 4])
print(x.sum(dim=(0, 1)))
print(x.mean(dim=(0, 1)))
print(x.view(x.size(0), -1).shape)
