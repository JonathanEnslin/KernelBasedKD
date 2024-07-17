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
from utils.early_stopping import EarlyStopping

from training_utils.training_step import TrainStep
from training_utils.validation_step import ValidationStep
from training_utils.testing_step import TestStep

def main():
    parser = argparse.ArgumentParser(description="Training script for NN")
    parser.add_argument('--params', type=str, required=True, help='Path to the params.json file')
    parser.add_argument('--param_set', type=str, required=True, help='Name of the parameter set to use')
    parser.add_argument('--resume', type=str, default='', help='Path to the checkpoint file to resume training')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (resnet20, resnet56, resnet110...)')
    parser.add_argument('--run_name', type=str, default='', help='Optional run name to overwrite the generated name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Frequency of checkpoint saving in epochs')
    parser.add_argument('--use_val', action='store_true', help='Use a validation set')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of training data used for validation set (0 < val_size < 1)')
    parser.add_argument('--disable_test', action='store_true', help='Disable the test set if validation set is used')
    parser.add_argument('--csv_dir', type=str, default='csv_logs', help='Directory to save CSV logs')
    parser.add_argument('--early_stopping_patience', type=int, help='Patience for early stopping')
    parser.add_argument('--early_stopping_start_epoch', type=int, help='Start early stopping after this many epochs')
    parser.add_argument('--dataset_dir', type=str, default='data', help='Directory of the dataset (default is data)')
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100'], help='Dataset to use (CIFAR10 or CIFAR100)')
    parser.add_argument('--model_save_dir', type=str, default='trained_models', help='Directory to save trained models')
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
        run_name = args.run_name or os.path.basename(args.resume).split('_epoch')[0]

    print(f"Using run name: {run_name}")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.exists(args.csv_dir):
        os.makedirs(args.csv_dir)

    csv_file = os.path.join(args.csv_dir, f"{run_name}_metrics.csv")

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
        "dataset_dir": args.dataset_dir,
        "dataset": args.dataset,
        "device": str(device)
    }
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    # Define transformations for the training, validation, and test sets
    if args.dataset == 'CIFAR10':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_class = torchvision.datasets.CIFAR10
    else:  # CIFAR100
        num_classes = 100
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
        dataset_class = torchvision.datasets.CIFAR100

    # Load the dataset
    dataset = dataset_class(root=args.dataset_dir, train=True, download=True, transform=transform_train)
    
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
        testset = dataset_class(root=args.dataset_dir, train=False, download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=params['training']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    # Initialize the nn model
    model = initialize_model(args.model_name, num_classes=num_classes, device=device)

    # Define the start time for the logger
    start_time = datetime.now()

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

    num_epochs = params['training']['max_epochs']
    start_epoch = 0

    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, schedulers, scaler, filename=args.resume)

    train_step = TrainStep(model, trainloader, criterion, optimizer, scaler, schedulers, device, writer, csv_file, start_time)
    if args.use_val:
        validation_step = ValidationStep(model, valloader, criterion, device, writer, csv_file, start_time, early_stopping)
    if not args.use_val or not args.disable_test:
        test_step = TestStep(model, testloader, criterion, device, writer, csv_file, start_time)

    for epoch in range(start_epoch, num_epochs):
        train_step(epoch)
        if args.use_val:
            if validation_step(epoch):
                break  # Early stopping triggered, exit training loop
        if not args.use_val or not args.disable_test:
            test_step(epoch)

        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_filename = os.path.join(args.checkpoint_dir, f"{run_name}_epoch{epoch+1}.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'schedulers': [scheduler.state_dict() for scheduler in schedulers],
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=checkpoint_filename)

    model.save(f'./{args.model_save_dir}/{run_name}.pth')
    writer.close()

if __name__ == "__main__":
    main()
