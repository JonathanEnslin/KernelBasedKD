import torchvision.datasets;
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

# A class that extends the CIFAR100 dataset class, but getitem also returns the index of the image
class IndexedCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index
    

class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class IndexedTinyImageNet(Dataset):
    def __init__(self, root_dir: Path, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.data = []
        self.labels = []

        # Load images and labels
        self._load_data()

    def _load_data(self):
        # Train/Val/Test directories
        dataset_split_dir = self.root_dir / self.split

        # Iterate over class directories
        for label_dir in dataset_split_dir.glob("*"):
            class_images_dir = label_dir / "images"
            for img_path in class_images_dir.glob("*.JPEG"):
                self.data.append(img_path)
                self.labels.append(label_dir.name)

        # Create a mapping of class names to class indices
        self.label_to_index = {label: i for i, label in enumerate(sorted(set(self.labels)))}
        self.indexed_labels = [self.label_to_index[label] for label in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.indexed_labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label, idx  # Returning image, label, and index


if __name__ == "__main__":
    import torch
    # Load the CIFAR100 dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = IndexedCIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = IndexedCIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False
    )

    # Print the dataset size
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Print the dataloader size
    print(f"Train dataloader size: {len(train_loader)}")
    print(f"Test dataloader size: {len(test_loader)}")

    for i, (images, labels, indices) in enumerate(train_loader):
        print(f"Batch {i+1}: {images.shape}, {labels.shape}, {indices}")
        if i == 0:
            break
