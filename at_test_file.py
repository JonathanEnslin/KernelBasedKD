import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.resnet import resnet20  # Assuming ResNet20 is defined in models.resnet

# Constants and Hyperparameters
NUM_CLASSES = 100
BATCH_SIZE = 128
TEST_BATCH_SIZE = 100
NUM_EPOCHS = 150
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_SCHEDULER_T_MAX = NUM_EPOCHS  # For CosineAnnealingLR
PRINT_INTERVAL = 100  # Batches after which to print training status
NUM_WORKERS = 4  # Number of subprocesses to use for data loading

# Check if CUDA is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data augmentation and normalization for training
TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                         std=[0.2673, 0.2564, 0.2762]),
])

# Normalization for testing
TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                         std=[0.2673, 0.2564, 0.2762]),
])

# Load CIFAR-100 datasets
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=TRANSFORM_TRAIN)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)

testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=TRANSFORM_TEST)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)

# Initialize the ResNet20 model
net = resnet20(num_classes=NUM_CLASSES)
net.to(DEVICE)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LR_SCHEDULER_T_MAX)

# Training function
def train(epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % PRINT_INTERVAL == 0 or batch_idx == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(trainloader)}], '
                  f'Loss: {running_loss/(batch_idx+1):.4f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')

# Testing function
def test(epoch):
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Test Loss: {test_loss/len(testloader):.4f}, '
          f'Test Accuracy: {100.*correct/total:.2f}%')

if __name__ == '__main__':
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train(epoch)
        test(epoch)
        scheduler.step()
