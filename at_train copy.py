import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
import argparse
import os
from datetime import datetime

from utils.checkpoint_utils import save_checkpoint, load_checkpoint
import utils.miscellaneous
from utils.param_utils import load_params
from utils.model_utils import initialize_model, get_optimizer, get_schedulers
from training_utils.early_stopping import EarlyStopping
from utils.best_model_tracker import BestModelTracker
from utils.data.dataset_getters import get_cifar100_transforms, get_cifar10_transforms, get_dataset_info
import utils.config_utils as config_utils

from training_utils.training_step import TrainStep
from training_utils.validation_step import ValidationStep
from training_utils.testing_step import TestStep

from loss_functions.AttentionTransfer.attention_transfer import ATLoss
from models.resnet import resnet56
from utils.data.indexed_dataset import IndexedCIFAR10, IndexedCIFAR100
from utils.teacher.teacher_model_handler import TeacherModelHandler
from utils.amp_grad_scaling_handler import get_amp_and_grad_scaler


import utils

def fmt_duration(duration):
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60
    ms = (duration - int(duration)) * 1000
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s {int(ms)}ms"


def main():
    parser = argparse.ArgumentParser(description="Training script for NN")
    parser.add_argument('--params', type=str, required=True, help='Path to the params.json file')
    parser.add_argument('--param_set', type=str, required=True, help='Name of the parameter set to use')
    parser.add_argument('--resume', type=str, default='', help='Path to the checkpoint file to resume training')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (resnet20, resnet56, resnet110...)')
    parser.add_argument('--run_name', type=str, default='', help='Optional run name to overwrite the generated name')
    parser.add_argument('--checkpoint_dir', type=str, default='run_data/checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Frequency of checkpoint saving in epochs')
    parser.add_argument('--use_val', action='store_true', help='Use a validation set')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of training data used for validation set (0 < val_size < 1)')
    parser.add_argument('--disable_test', action='store_true', help='Disable the test set if validation set is used')
    parser.add_argument('--csv_dir', type=str, default='run_data/csv_logs', help='Directory to save CSV logs')
    parser.add_argument('--early_stopping_patience', type=int, help='Patience for early stopping')
    parser.add_argument('--early_stopping_start_epoch', type=int, help='Start early stopping after this many epochs')
    parser.add_argument('--dataset_dir', type=str, default='data', help='Directory of the dataset (default is data)')
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100'], help='Dataset to use (CIFAR10 or CIFAR100)')
    parser.add_argument('--model_save_dir', type=str, default='run_data/trained_models', help='Directory to save trained models')
    parser.add_argument('--device', type=str, default='cuda', choices=["cpu", "cuda"] , help='Device to use (cpu or cuda)')
    parser.add_argument('--track_best_after_epoch', type=int, default=10, help='Number of epochs to wait before starting to track the best model (Only enabled when using validation set)')
    parser.add_argument('--val_split_random_state', type=int, default=None, help='Random state for the validation split')
    parser.add_argument('--use_split_indices_from_file', type=str, default=None, help='Path to a file containing indices for the train and validation split')
    parser.add_argument('--disable_auto_run_indexing', action='store_true', help='Disable automatic run indexing (i.e. _run1, _run2, etc.)')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision (AMP) and Gradient Scaling (Warning: may cause gradient under/overflow, use carefully)')
    parser.add_argument('--at_mode', type=str, default='impl', choices=['impl', 'paper', 'zoo'], help='Mode for attention transfer loss')
    parser.add_argument('--at_beta', type=float, default=1.0, help='Beta parameter for attention transfer loss, default None means will be inferred automatically based on mode')
    parser.add_argument('--use_cached_teacher', action='store_true', help='Use cached teacher logits and feature maps')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for the DataLoader')
    parser.add_argument('--disable_persistent_workers', action='store_true', help='Disables using persistent workers for the DataLoader')
    args = parser.parse_args()

    # Initialise logger
    logger = print

    # Load the training parameters
    params = load_params(args.params, args.param_set)

    # Check that model_save_dir, checkpoint_dir, and csv_dir are valid directories
    # if they do not exist, create them, if they can not be created print an error message and exit gracefully
    if not utils.miscellaneous.ensure_dir_existence([args.model_save_dir, args.checkpoint_dir, args.csv_dir]):
        exit(1)

    # Set the device
    requested_device = args.device
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    logger(device)

    # Get gradscaler and autocast context class
    scaler, autocast = get_amp_and_grad_scaler(args, device, logger=logger)


    # Get the run name
    run_name = config_utils.get_run_name(args)

    # if validation set is not used, print that track best and early stopping are disabled (if they are specified)
    if not args.use_val:
        if args.early_stopping_patience is not None:
            logger("Warning: Early stopping is disabled because validation set is not used.")
        elif args.early_stopping_start_epoch is not None:
            logger("Warning: Early stopping is disabled because validation set is not used.")
        if args.track_best_after_epoch is not None:
            logger("Warning: Best model tracking is disabled because validation set is not used.")

    logger(f"-------> Using run name: {run_name}")

    csv_file = os.path.join(args.csv_dir, f"{run_name}_metrics.csv")

    config_utils.print_config(params, run_name, args, device, logger=print)

    # Get the dataset info
    dataset_class, num_classes, transform_train, transform_test = get_dataset_info(args.dataset)
    if dataset_class is None:
        exit(1)

    # Load the dataset
    train_dataset = dataset_class(root=args.dataset_dir, train=True, download=True, transform=transform_train)

    trainloader, valloader, testloader, val_split_random_state \
            = config_utils.get_data_loaders(args, params, train_dataset, run_name, transform_train, transform_test, dataset_class)

    # Initialize the nn model
    model = initialize_model(args.model_name, num_classes=num_classes, device=device)

    # Load and cache teacher model data
    logger("Setting up teacher model")
    teacher_model_handler = TeacherModelHandler(model_class=resnet56,
                                                teacher_file_name="resnet56_params3_CIFAR100_1_00025.pth",
                                                device=device,
                                                num_classes=num_classes,
                                                printer=print)
    
    teacher = teacher_model_handler.load_teacher_model()
    if teacher is None:
        exit(1)

    # Default to None, will be set to the logits and feature maps if using cached teacher
    teacher_logits = None
    teacher_layer_groups_preactivation_fmaps = None
    teacher_layer_groups_post_activation_fmaps = None   
    if args.use_cached_teacher:
        teacher_logits, teacher_layer_groups_preactivation_fmaps, teacher_layer_groups_post_activation_fmaps = \
            teacher_model_handler.generate_and_save_teacher_logits_and_feature_maps(trainloader=trainloader, train_dataset=train_dataset)        
        
    del teacher_model_handler # to release references so that memory can be cleared up later
    logger("Teacher model setup completed.")    
    logger("")

    # Set up loss functions
    criterion = ATLoss(student=model,
                        teacher=teacher,
                        beta=args.at_beta, mode=args.at_mode, device=device,
                        cached_post_activation_fmaps=teacher_layer_groups_post_activation_fmaps,
                        cached_pre_activation_fmaps=teacher_layer_groups_preactivation_fmaps,
                        use_post_activation=None)
    del teacher_layer_groups_post_activation_fmaps
    del teacher_layer_groups_preactivation_fmaps

    test_val_criterion = nn.CrossEntropyLoss()    

    # Restore hook states to their ideal values
    teacher.set_hook_device_state(args.device if torch.cuda.is_available() else "cpu")
    model.set_hook_device_state(args.device if torch.cuda.is_available() else "cpu")

    # Define the optimizer and learning rate scheduler
    optimizer = get_optimizer(params, model)
    schedulers = get_schedulers(params, optimizer)

    # Initialize TensorBoard writer
    writer_name = config_utils.get_writer_name(kd_mode="AT", args=args, run_name=run_name)
    writer = SummaryWriter(writer_name)

    # Define the start time for the logger
    start_time = datetime.now()

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

    train_step = TrainStep(model, trainloader, criterion, optimizer, scaler, schedulers, device, writer, csv_file, start_time, autocast, is_kd=True)
    if args.use_val:
        validation_step = ValidationStep(model, valloader, test_val_criterion, device, writer, csv_file, start_time, autocast, early_stopping, best_model_tracker)
    if not args.use_val or not args.disable_test:
        test_step = TestStep(model, testloader, test_val_criterion, device, writer, csv_file, start_time, autocast)

    # Main training loop
    times_at_epoch_end = []
    start_time = datetime.now()
    prev_time = start_time
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = datetime.now()
        train_step(epoch)
        if args.use_val:
            early_stop = validation_step(epoch)
            if early_stop:
                break  # Early stopping triggered, exit training loop
        if not args.use_val or not args.disable_test:
            test_step(epoch)

        curr_time = datetime.now()
        times_at_epoch_end.append(curr_time - prev_time)
        avg_time_per_epoch = sum([dur.total_seconds() for dur in times_at_epoch_end[-20:]]) / len(times_at_epoch_end[-20:])
        prev_time = curr_time

        logger(f"Epoch {epoch+1} took {fmt_duration((curr_time - epoch_start_time).total_seconds())}, Total time: {fmt_duration((curr_time - start_time).total_seconds())}")
        logger(f"Average time per epoch: {fmt_duration(avg_time_per_epoch)}")
        logger(f"Estimated time remaining: {fmt_duration(avg_time_per_epoch * (num_epochs - epoch - 1))}")
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



