import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torch.optim.lr_scheduler import MultiStepLR
from models.resnet import resnet56

def fmt_duration(duration):
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60
    ms = (duration - int(duration)) * 1000
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s {int(ms)}ms"

# Configurable parameters
use_amp = False  # Set this to False to disable AMP

if use_amp:
    from torch.cuda.amp import GradScaler, autocast
else:
    # Dummy classes and contexts for compatibility when AMP is not used
    class GradScaler:
        def scale(self, loss):
            return loss
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

    class autocast:
        def __init__(self):
            pass
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_value, traceback):
            pass

if __name__ == '__main__':
    # Hyperparameters
    lr = 0.05
    momentum = 0.9
    nesterov = False
    weight_decay = 5e-4
    gamma = 0.1
    milestones = [150, 180, 210]
    batch_size = 256
    num_workers = 4
    num_epochs = 240  # Specify the number of epochs
    # lr = 0.1
    # momentum = 0.9
    # nesterov = True
    # weight_decay = 0.0005
    # gamma = 0.2
    # milestones = [60, 80, 90]
    # batch_size = 128
    # num_workers = 2
    # num_epochs = 110  # Specify the number of epochs

    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Datasets and dataloaders
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    # Model
    model = resnet56(num_classes=100)
    model.set_hook_device_state("same")

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Print the model architecture
    print(model)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)

    # Scheduler
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # GradScaler for automatic mixed precision
    scaler = GradScaler()

    # Training function
    def train(epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:  # Print every 100 batches
                print(f'Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(trainloader)}], Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')

        scheduler.step()

    epoch_durs = []

    # Testing function
    def test():
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print(f'Test Loss: {test_loss/len(testloader):.4f}, \n===============> Test Accuracy: {100.*correct/total:.2f}%')

    # Training loop
    prev_time = time.time()
    for epoch in range(num_epochs):
        train(epoch)
        if epoch % 10 == 0 or epoch > 88:
            test()
        epoch_durs.append((time.time() - prev_time))
        prev_time = time.time()
        avg_time = sum(epoch_durs[-10:]) / len(epoch_durs[-10:])
        print(f"Epoch {epoch+1} duration: {fmt_duration(epoch_durs[-1])}, Average duration: {fmt_duration(avg_time)}")
        print(f"Estimated time remaining: {fmt_duration(avg_time * (num_epochs - epoch - 1))}")

    torch.save(model.state_dict(), 'resnet56_cifar100_the_last_hope_3_oof.pth')
