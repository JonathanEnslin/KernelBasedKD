import torchvision.datasets;

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
