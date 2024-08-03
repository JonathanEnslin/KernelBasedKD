# A script that loads the CIFAR100 dataset

# import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# Load the CIFAR100 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=128,
    shuffle=False
)

# Print the dataset size
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Print the dataloader size
print(f"Train dataloader size: {len(train_loader)}")
print(f"Test dataloader size: {len(test_loader)}")

for i, (images, labels) in enumerate(train_loader):
    print(f"Batch {i+1}: {images.shape}, {labels.shape}")
    if i == 0:
        break