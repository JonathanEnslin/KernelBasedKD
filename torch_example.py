# Obtained from:
# https://pyhopper.readthedocs.io/en/latest/examples/torch_cifar10.html

import numpy as np
import torchvision
import torch
import torch.utils.data
from torch.utils.data import SubsetRandomSampler
from tqdm.auto import tqdm
from models.resnet import resnet20

import pyhopper


def get_cifar_loader(batch_size, erasing_prob, for_validation):
    mean = np.array([125.30691805, 122.95039414, 113.86538318]) / 255.0
    std = np.array([62.99321928, 62.08870764, 66.70489964]) / 255.0

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomErasing(p=erasing_prob),
            torchvision.transforms.Normalize(mean, std),
        ]
    )
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )

    dataset_dir = r"C:\Users\jonat\OneDrive\UNIV stuff\CS4\COS700\Dev\KernelBasedKD\data"
    train_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=True, transform=train_transform, download=True
    )
    train_sampler, test_sampler = torch.utils.data.RandomSampler(train_dataset), None
    if for_validation:
        test_dataset = torchvision.datasets.CIFAR10(
            dataset_dir, train=True, transform=test_transform, download=True
        )
        indices = np.random.default_rng(12345).permutation(len(train_dataset))
        valid_size = int(0.05 * len(train_dataset))
        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(valid_idx)
    else:
        test_dataset = torchvision.datasets.CIFAR10(
            dataset_dir, train=False, transform=test_transform, download=True
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    return train_loader, test_loader


def training_epoch(model, optimizer, scheduler, loss_fn, train_loader, for_validation):
    model.train()

    correct_samples = 0
    num_samples = 0
    total_loss = 0
    prog_bar = tqdm(total=len(train_loader), disable=not for_validation)
    for step, (data, targets) in enumerate(train_loader):
        data = data.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()
        with torch.no_grad():
            _, preds = torch.max(outputs, dim=1)
            correct_samples += preds.eq(targets).sum().item()
            num_samples += data.size(0)
        loss.backward()
        optimizer.step()
        scheduler.step()

        prog_bar.update(1)
        prog_bar.set_description_str(
            f"loss={total_loss/(step+1):0.3f}, train_acc={100*correct_samples/num_samples:0.2f}%"
        )
    prog_bar.close()


def evaluate(model, data_loader):
    model.eval()

    with torch.no_grad():
        num_samples = 0
        correct_samples = 0

        for step, (data, targets) in enumerate(data_loader):
            data = data.cuda()
            targets = targets.cuda()

            outputs = model(data)
            _, preds = torch.max(outputs, dim=1)
            correct_samples += preds.eq(targets).sum().item()
            num_samples += data.size(0)

    return float(correct_samples / num_samples)


def train_cifar10(params, for_validation=True):
    model = resnet20(num_classes=10)
    model.set_hook_device_state('same')
    model = model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params["lr"],
        momentum=0.9,
        weight_decay=params["weight_decay"],
        nesterov=True,
    )

    train_loader, val_loader = get_cifar_loader(
        128, params["erasing_prob"], for_validation
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader) * 100, eta_min=params["eta_min"]
    )

    for e in range(3):
        training_epoch(
            model, optimizer, scheduler, loss_fn, train_loader, for_validation
        )
        if not for_validation:
            val_acc = evaluate(model, val_loader)
            print(f"epoch {e} val_acc={100*val_acc:0.2f}%")

    return evaluate(model, val_loader)


if __name__ == "__main__":
    search = pyhopper.Search(
        {
            "lr": pyhopper.float(0.5, 0.05, precision=1, log=True),
            "eta_min": pyhopper.choice([0, 1e-4, 1e-3, 1e-2], is_ordinal=True),
            "weight_decay": pyhopper.float(1e-6, 1e-2, log=True, precision=1),
            "erasing_prob": pyhopper.float(0, 1, precision=1),
        }
    )
    best_params = search.run(
        train_cifar10,
        direction="max",
        runtime="1m",
        n_jobs="per-gpu",
    )
    test_acc = train_cifar10(best_params, for_validation=False)
    print(best_params)
    print(type(best_params))
    print(f"Tuned params: Test accuracy = {100 * test_acc}")
