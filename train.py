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

from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.param_utils import load_params
from utils.model_utils import initialize_model, get_optimizer, get_schedulers
from utils.early_stopping import EarlyStopping
from utils.best_model_tracker import BestModelTracker
from utils.dataset_getters import get_cifar100_transforms, get_cifar10_transforms
import utils.config_utils as config_utils
import utils.data.dataset_splitter as dataset_splitter

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
    parser.add_argument('--device', type=str, default='cuda', choices=["cpu", "cuda"] , help='Device to use (cpu or cuda)')
    parser.add_argument('--track_best_after_epoch', type=int, default=10, help='Number of epochs to wait before starting to track the best model (Only enabled when using validation set)')
    parser.add_argument('--val_split_random_state', type=int, default=None, help='Random state for the validation split')
    parser.add_argument('--use_split_indices_from_file', type=str, default=None, help='Path to a file containing indices for the train and validation split')
    parser.add_argument('--disable_auto_run_indexing', action='store_true', help='Disable automatic run indexing (i.e. _run1, _run2, etc.)')
    args = parser.parse_args()

    params = load_params(args.params, args.param_set)

    # Check that model_save_dir, checkpoint_dir, and csv_dir are valid directories
    # if they do not exist, create them, if they can not be created print an error message and exit gracefully
    for directory in [args.model_save_dir, args.checkpoint_dir, args.csv_dir]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                print(f"Error: {directory} could not be created. {e}")
                exit(1)


    # Set the device
    requested_device = args.device
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    print(device)

    # Get the run name
    run_name = config_utils.get_run_name(args)

    # if validation set is not used, print that track best and early stopping are disabled (if they are specified)
    if not args.use_val:
        if args.early_stopping_patience is not None:
            print("Warning: Early stopping is disabled because validation set is not used.")
        elif args.early_stopping_start_epoch is not None:
            print("Warning: Early stopping is disabled because validation set is not used.")
        if args.track_best_after_epoch is not None:
            print("Warning: Best model tracking is disabled because validation set is not used.")

    print(f"Using run name: {run_name}")

    csv_file = os.path.join(args.csv_dir, f"{run_name}_metrics.csv")

    config_utils.print_config(params, run_name, args, device, printer=print)

    # Define transformations for the training, validation, and test sets
    if args.dataset == 'CIFAR10':
        num_classes = 10
        transform_train, transform_test = get_cifar10_transforms()
        dataset_class = torchvision.datasets.CIFAR10
    else:  # CIFAR100
        num_classes = 100
        transform_train, transform_test = get_cifar100_transforms()
        dataset_class = torchvision.datasets.CIFAR100

    # Load the dataset
    dataset = dataset_class(root=args.dataset_dir, train=True, download=True, transform=transform_train)

    trainloader, valloader, testloader, val_split_random_state \
            = config_utils.get_data_loaders(args, params, dataset, run_name, transform_train, transform_test, dataset_class)

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
    writer = SummaryWriter(f"runs/{args.dataset}/{run_name}")

    # Initialize EarlyStopping if validation is used and early stopping params are passed
    early_stopping = None
    if args.use_val and (args.early_stopping_patience is not None or args.early_stopping_start_epoch is not None):
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience or 10, 
            verbose=True, 
            enabled_after_epoch=args.early_stopping_start_epoch or 10,
            monitor='loss'
        )

    # Initialize BestModelTracker if validation is used
    best_model_tracker = None
    if args.use_val:
        best_model_tracker = BestModelTracker(
            verbose=True,
            delta=0,
            path=os.path.join(args.model_save_dir, f"{run_name}_best.pth"),
            monitor='loss', # Only loss is currently supported
            enabled_after_epoch=args.track_best_after_epoch or 10
        )


    num_epochs = params['training']['max_epochs']
    start_epoch = 0

    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, schedulers, scaler, filename=args.resume)

    train_step = TrainStep(model, trainloader, criterion, optimizer, scaler, schedulers, device, writer, csv_file, start_time)
    if args.use_val:
        validation_step = ValidationStep(model, valloader, criterion, device, writer, csv_file, start_time, early_stopping, best_model_tracker)
    if not args.use_val or not args.disable_test:
        test_step = TestStep(model, testloader, criterion, device, writer, csv_file, start_time)

    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        train_step(epoch)
        if args.use_val:
            early_stop = validation_step(epoch)
            if early_stop:
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
                'val_split_random_state': val_split_random_state,
            }, is_best=False, filename=checkpoint_filename)

    model.save(f'./{args.model_save_dir}/{run_name}.pth')
    writer.close()

if __name__ == "__main__":
    main()
