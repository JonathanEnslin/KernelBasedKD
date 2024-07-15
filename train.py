import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
import argparse
import os
from datetime import datetime
from sklearn.metrics import f1_score

from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.param_utils import load_params
from utils.model_utils import initialize_model, get_optimizer, get_schedulers
from utils.log_utils import log_to_csv, create_log_entry
from utils.early_stopping import EarlyStopping  # Import the EarlyStopping class

def main():
    parser = argparse.ArgumentParser(description="Training script for CIFAR-100 with ResNet")
    parser.add_argument('--params', type=str, required=True, help='Path to the params.json file')
    parser.add_argument('--param_set', type=str, required=True, help='Name of the parameter set to use')
    parser.add_argument('--resume', type=str, default='', help='Path to the checkpoint file to resume training')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (resnet20, resnet56, resnet110)')
    parser.add_argument('--run_name', type=str, default='', help='Optional run name to overwrite the generated name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Frequency of checkpoint saving in epochs')
    parser.add_argument('--use_val', action='store_true', help='Use a validation set')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of training data used for validation set (0 < val_size < 1)')
    parser.add_argument('--disable_test', action='store_true', help='Disable the test set if validation set is used')
    parser.add_argument('--csv_dir', type=str, default='csv_logs', help='Directory to save CSV logs')
    parser.add_argument('--early_stopping_patience', type=int, help='Patience for early stopping')
    parser.add_argument('--early_stopping_start_epoch', type=int, help='Start early stopping after this many epochs')
    args = parser.parse_args()

    params = load_params(args.params, args.param_set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate or use provided run name
    run_name_base = args.run_name or f"{args.model_name}_{args.param_set}"
    run_name = run_name_base + "_run1"
    run_counter = 2

    while os.path.exists(f"runs/{run_name}") or os.path.exists(f"{run_name}.pth"):
        run_name = f"{run_name_base}_run{run_counter}"
        run_counter += 1

    # If resuming training, use the run name from the checkpoint file or the provided run name
    if args.resume:
        # Use the run from args or otherwise use the run name from the checkpoint file
        run_name = args.run_name or os.path.basename(args.resume).split('_epoch')[0]

    print(f"Using run name: {run_name}")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.exists(args.csv_dir):
        os.makedirs(args.csv_dir)

    csv_file = os.path.join(args.csv_dir, f"{run_name}_metrics.csv")
    start_time = datetime.now()

    # Print out the configuration
    config = {
        "params": params,
        "run_name": run_name,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_freq": args.checkpoint_freq,
        "use_val": args.use_val,
        "val_size": args.val_size,
        "disable_test": args.disable_test,
        "csv_dir": args.csv_dir,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_start_epoch": args.early_stopping_start_epoch,
        "device": str(device)
    }
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

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
    
    if args.use_val:
        val_size = int(args.val_size * len(dataset))
        train_size = len(dataset) - val_size
        trainset, valset = random_split(dataset, [train_size, val_size])
    else:
        trainset = dataset

    trainloader = DataLoader(trainset, batch_size=params['training']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    
    if args.use_val:
        valloader = DataLoader(valset, batch_size=params['training']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    if not args.use_val or not args.disable_test:
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=params['training']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    # Initialize the nn model
    model = initialize_model(args.model_name, device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(params, model)
    scaler = amp.GradScaler()
    schedulers = get_schedulers(params, optimizer)

    # Initialize TensorBoard writer
    writer = SummaryWriter(f"runs/{run_name}")

    # Initialize EarlyStopping if validation is used and early stopping params are passed
    early_stopping = None
    if args.use_val and (args.early_stopping_patience is not None or args.early_stopping_start_epoch is not None):
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience or 10, 
            verbose=True, 
            path=os.path.join(args.checkpoint_dir, f"{run_name}_best.pth"),
            enabled_after_epoch=args.early_stopping_start_epoch or 10,
            monitor='loss'
        )

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
        
        # Step the schedulers
        for scheduler in schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(running_loss)
            else:
                scheduler.step()

        # Calculate metrics
        epoch_loss = running_loss / len(trainloader)
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')
        writer.add_scalar('training_epoch_loss', epoch_loss, epoch)
        writer.add_scalar('training_accuracy', accuracy, epoch)
        writer.add_scalar('training_f1_score', f1, epoch)

        log_entry = create_log_entry(epoch, 'train', epoch_loss, accuracy, f1, start_time)
        log_to_csv(csv_file, log_entry)

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
        # print the validation accuracy
        print(f'--> Validation accuracy: {accuracy:.2f}%')

        log_entry = create_log_entry(epoch, 'validation', epoch_loss, accuracy, f1, start_time)
        log_to_csv(csv_file, log_entry)

        # Early stopping check
        if early_stopping:
            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                return True
        return False

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
        # print the test accuracy
        print(f'--> Test accuracy: {accuracy:.2f}%')

        log_entry = create_log_entry(epoch, 'test', epoch_loss, accuracy, f1, start_time)
        log_to_csv(csv_file, log_entry)

    num_epochs = params['training']['max_epochs']
    start_epoch = 0

    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, schedulers, scaler, filename=args.resume)

    for epoch in range(start_epoch, num_epochs):
        train(epoch)
        if args.use_val:
            if validate(epoch):
                break  # Early stopping triggered, exit training loop
        if not args.use_val or not args.disable_test:
            test(epoch)

        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_filename = os.path.join(args.checkpoint_dir, f"{run_name}_epoch{epoch+1}.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'schedulers': [scheduler.state_dict() for scheduler in schedulers],
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=checkpoint_filename)

    model.save(f'./trained_models/{run_name}.pth')
    writer.close()

if __name__ == "__main__":
    main()
