import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
import models.base_model as base_model

from models.resnet import resnet20, resnet56, resnet110  # Custom ResNet for CIFAR datasets
import os

RUN_NAME = 'resnet20_cifar100_params4_post_activation_run1'
CHECKPOINT_NAME = f'{RUN_NAME}.pth.tar'

def save_checkpoint(state, is_best, filename=CHECKPOINT_NAME):
    print(f"=> Saving checkpoint '{filename}'. Please do not interrupt the saving process else the checkpoint file might get corrupted.")
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'model_best.pth.tar')
    print(f"=> Checkpoint saved at '{filename}'")
    

def load_checkpoint(model, optimizer, scheduler, scaler, filename=CHECKPOINT_NAME):
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return start_epoch
    else:
        print(f"=> no checkpoint found at '{filename}'")
        return 0

if __name__ == "__main__":

    # Define transformations for the training and test sets
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

    # Load the CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize the nn model
    model: base_model = resnet20()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scaler = amp.GradScaler()

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)

    # Initialize TensorBoard writer
    writer = SummaryWriter(f"runs/{RUN_NAME}")

    # Training function
    def train(epoch):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            if i % 100 == 99:  # Log every 100 mini-batches
                print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.3f}')
                writer.add_scalar('training_loss', running_loss / 100, epoch * len(trainloader) + i)
                running_loss = 0.0
        
        # Step the scheduler
        scheduler.step()

    # Test function
    def test(epoch):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with amp.autocast():
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')
        writer.add_scalar('test_accuracy', accuracy, epoch)

    # Main training loop
    num_epochs = 240
    start_epoch = load_checkpoint(model, optimizer, scheduler, scaler, filename=CHECKPOINT_NAME)

    for epoch in range(start_epoch, num_epochs):
        train(epoch)
        test(epoch)

        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False)


    model.save(f'{RUN_NAME}.pth')
    # Close the TensorBoard writer
    writer.close()
