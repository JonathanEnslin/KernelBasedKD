import torchvision
import torchvision.transforms as transforms

from utils.data.dataset_splitter import *

dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())

# train_set, val_set = split_dataset(dataset, test_size=0.2, stratify=True, random_state=None, save_to_file='indices-2.json')
train_set, val_set = split_dataset_from_file(dataset, 'indices-2.json')

# print the forst 10 indices of the training set
print(train_set.indices[:10])
# print the forst 10 indices of the validation set
print(val_set.indices[:10])
