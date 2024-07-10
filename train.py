import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
import argparse
import os
import csv
from sklearn.metrics import f1_score

from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.param_utils import load_params
from utils.model_utils import initialize_model, get_optimizer, get_scheduler

def log_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def main():
    parser = argparse.ArgumentParser(description="Training script for CIFAR-100 with ResNet")
    parser.add_argument('--params', type=str, required=True, help='Path to the params.json file')
    parser.add_argument('--param_set', type=str, required=True, help='Name of the parameter set to use')
    parser.add_argument('--resume', type=str, default='', help='Path to the checkpoint file to resume training')
    parser.add_argument('--model_name', type=str, default='resnet20', help='Model name (resnet20, resnet56, resnet110)')
    parser.add_argument('--run_name', type=str, default='', help='Optional run name to overwrite the generated name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Frequency of checkpoint saving in epochs')
    args = parser.parse_args()

    params = load_params(args.params, args.param_set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate or use provided run name
    run_name_base = args.run_name or f"{args.param_set}_{args.model_name}"
    run_name = run_name_base
    run_counter = 1

    while os.path.exists(f"runs/{run_name}") or os.path.exists(f"{run_name}.pth"):
        run_name = f"{run_name_base}_run{run_counter}"
        run_counter += 1

    print(f"Using run name: {run_name}")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    csv_file = f"runs/{run_name}_metrics.csv"

    # Define transformations for the training, validation, and test sets
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
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=params['training']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    valloader = DataLoader(valset, batch_size=params['training']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=params['training']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    # Initialize the nn model
    model = initialize_model(args.model_name, device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(params, model)
    scaler = amp.GradScaler()
    scheduler = get_scheduler(params, optimizer)

    # Initialize TensorBoard writer
    writer = SummaryWriter(f"runs/{run_name}")

    # Training function
    def train(epoch):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
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
            
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            if i % 100 == 99:  # Log every 100 mini-batches
                print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.3f}')
                writer.add_scalar('training_loss', running_loss / 100, epoch * len(trainloader) + i)
                running_loss = 0.0
        
        scheduler.step()

        # Calculate metrics
        epoch_loss = running_loss / len(trainloader)
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')
        writer.add_scalar('training_epoch_loss', epoch_loss, epoch)
        writer.add_scalar('training_accuracy', accuracy, epoch)
        writer.add_scalar('training_f1_score', f1, epoch)

        log_to_csv(csv_file, {
            'epoch': epoch,
            'phase': 'train',
            'loss': epoch_loss,
            'accuracy': accuracy,
            'f1_score': f1
        })

    def validate(epoch):
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(valloader)
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')
        writer.add_scalar('validation_loss', epoch_loss, epoch)
        writer.add_scalar('validation_accuracy', accuracy, epoch)
        writer.add_scalar('validation_f1_score', f1, epoch)

        log_to_csv(csv_file, {
            'epoch': epoch,
            'phase': 'validation',
            'loss': epoch_loss,
            'accuracy': accuracy,
            'f1_score': f1
        })

    def test(epoch):
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(testloader)
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')
        writer.add_scalar('test_loss', epoch_loss, epoch)
        writer.add_scalar('test_accuracy', accuracy, epoch)
        writer.add_scalar('test_f1_score', f1, epoch)

        log_to_csv(csv_file, {
            'epoch': epoch,
            'phase': 'test',
            'loss': epoch_loss,
            'accuracy': accuracy,
            'f1_score': f1
        })

    num_epochs = params['training']['max_epochs']
    start_epoch = 0

    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, scaler, filename=args.resume)

    for epoch in range(start_epoch, num_epochs):
        train(epoch)
        validate(epoch)
        test(epoch)

        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_filename = os.path.join(args.checkpoint_dir, f"{run_name}_epoch{epoch+1}.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=checkpoint_filename)

    model.save(f'{run_name}.pth')
    writer.close()

if __name__ == "__main__":
    main()
