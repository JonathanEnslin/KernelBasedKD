import torchvision.transforms as transforms
import utils.data.indexed_dataset as index_datasets

def get_cifar100_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    return train_transform, test_transform


def get_cifar10_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return train_transform, test_transform

def get_dataset_info(dataset_name, logger=print):
    dataset_class = None
    num_classes = None
    transform_train = None
    transform_test = None
    if dataset_name == 'CIFAR10':
        num_classes = 10
        transform_train, transform_test = get_cifar10_transforms()
        dataset_class = index_datasets.IndexedCIFAR10
    elif dataset_name == 'CIFAR100':
        num_classes = 100
        transform_train, transform_test = get_cifar100_transforms()
        dataset_class = index_datasets.IndexedCIFAR100
    else:
        logger(f"Uknown dataset: {dataset_name}")


    return dataset_class, num_classes, transform_train, transform_test

