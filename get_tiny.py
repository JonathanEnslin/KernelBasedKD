from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data.indexed_dataset import IndexedTinyImageNet

# Define transformations (resize to 64x64 since that's the original Tiny ImageNet resolution)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
root_dir = r'f:\Development\UNIV Dev\CS4\COS700\Datasets\tiny-64'
# Initialize the dataset
train_dataset = IndexedTinyImageNet(root=Path(root_dir), train=True, transform=transform)
val_dataset = IndexedTinyImageNet(root=Path(root_dir), train=False, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Example: Iterate through a batch of data
# for images, labels, indices in train_loader:
#     print(f"Batch of images shape: {images.shape}")
#     print(f"Batch of labels: {labels}")
#     print(f"Batch of indices: {indices}")
    # break  # Just one batch for demonstration


print(len(train_dataset), 0.05*len(train_dataset), 0.05*500)
print(len(val_dataset))