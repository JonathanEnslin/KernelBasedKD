import os
import sys
import multiprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyhopper
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from sklearn.model_selection import KFold
import numpy as np

import utils.data.indexed_dataset as index_datasets
import models.resnet as resnet_models  # Assuming resnet20 is defined in this module
from utils.data.dataset_getters import get_dataset_info
from utils.logger import Logger

from mock_args import Args as MockArgs

NAME = 'no_name_provided'

# Global variables

# PARAMS
FOLDS = 5  # Number of folds for K-fold cross-validation
PERCENTILE_PRUNE = 80
DATASET = "CIFAR100"
STUDENT_MODEL = "resnet20"
LOGGER_TAG = "base"
LOG_DIR = "hpar_tuning_logs/base"


folds = None
full_dataset = None
num_classes = None
shared_accuracies = None  # Will be initialized with a multiprocessing Manager

# Fixed parameters
optimizer_params = {
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 0.0005
}

scheduler_params = {
    "gamma": 0.2,
    "milestones": [60, 80, 90]
}

training_params = {
    "max_epochs": 2,
    "batch_size": 128
}

# Function to split the dataset into folds
def prepare_kfold_splits(dataset, n_splits=FOLDS):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=112)  # Added a fixed random_state for reproducibility
    return list(kf.split(dataset))  # Returns a list of (train_indices, val_indices) tuples

# Noisy objective function for K-Folds (Only takes param and eval_index)
def kfold_objective(param, eval_index):
    import time
    global folds, full_dataset, num_classes, device, shared_accuracies  # Access global variables
    global DATASET, STUDENT_MODEL, LOGGER_TAG, LOG_DIR, PERCENTILE_PRUNE

    args = MockArgs()
    args.dataset = DATASET
    args.model_name = STUDENT_MODEL
    args.param_set = 'tune'

    logger = Logger(args=args, log_to_file=True, data_dir=LOG_DIR, run_tag=LOGGER_TAG, teacher_type=None, kd_set=None)

    logger(f"Starting evaluation for Fold {eval_index + 1}")
    logger(f"Parameters:\n{param}")

    # Get the appropriate fold
    train_indices, val_indices = folds[eval_index]
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # Prepare data loaders
    train_loader = DataLoader(train_subset, batch_size=training_params["batch_size"], shuffle=True, drop_last=True, num_workers=0, persistent_workers=False)
    val_loader = DataLoader(val_subset, batch_size=training_params["batch_size"], shuffle=False)

    # Initialize the model and move it to the device
    start_time = time.time()
    model = resnet_models.resnet20(num_classes=num_classes).to(device)

    # Define loss and optimizer with the fixed parameters and searched lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=param['lr'],  # Use the learning rate from pyhopper search
        momentum=optimizer_params["momentum"],
        nesterov=optimizer_params["nesterov"],
        weight_decay=optimizer_params["weight_decay"]
    )

    # Scheduler (MultiStepLR)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=scheduler_params["milestones"],
        gamma=scheduler_params["gamma"]
    )

    # Training loop
    for epoch in range(training_params["max_epochs"]):
        logger(f"Fold {eval_index + 1}, Epoch {epoch + 1}")
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets, indices) in enumerate(train_loader):
            if batch_idx % 150 == 0:
                logger(f"Batch {batch_idx + 1}/{len(train_loader)}")
            
            # Move inputs and targets to the device
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        logger.log_to_csv({'phase': 'train', 'loss': train_loss / len(train_loader), "accuracy": -1.0})
        # Step the scheduler after each epoch
        scheduler.step()

    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, indices in val_loader:
            # Move inputs and targets to the device
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.log_to_csv({'phase': 'val', 'loss': val_loss / len(val_loader), "accuracy": correct / total})
    val_accuracy = 100. * correct / total
    logger(f"Fold {eval_index + 1}, Validation Accuracy: {val_accuracy:.2f}%")
    
    # Record the validation accuracy to the shared list
    shared_accuracies.append(val_accuracy)
    
    # Check if the current accuracy is in the lower P% of the results seen so far and prune if necessary
    if len(shared_accuracies) > 10 and eval_index != 0:  # Allow some evaluations before pruning
        threshold = np.percentile(shared_accuracies[:-eval_index-1], PERCENTILE_PRUNE)
        mean_config_accs = sum(shared_accuracies[-eval_index-1:]) / (eval_index + 1)
        if mean_config_accs < threshold:
            logger(f"Pruning evaluation for Fold {eval_index + 1}, Avg. Acc of Cur Conf {mean_config_accs:.2f}% below {PERCENTILE_PRUNE}th percentile")
            raise pyhopper.PruneEvaluation()

    end_time = time.time()
    logger(f"Fold {eval_index + 1}, Time taken: {end_time - start_time:.2f} seconds")
    return val_accuracy

# Main section to load dataset and perform K-fold splits once
def main(dataset_name="CIFAR10", requested_device="cuda"):
    global folds, full_dataset, num_classes, device, shared_accuracies  # Use global variables
    global NAME, FOLDS

    # Set the device based on the availability of CUDA
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get dataset info
    dataset_class, num_classes, transform_train, transform_test = get_dataset_info(dataset_name)
    
    # Load the dataset (indexed dataset with transforms)
    full_dataset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    
    # Perform K-Fold splitting once
    folds = prepare_kfold_splits(full_dataset, n_splits=FOLDS)

    # Initialize a multiprocessing-safe list for storing accuracies
    with multiprocessing.Manager() as manager:
        shared_accuracies = manager.list()

        # Define the parameter search space using pyhopper
        search: pyhopper.Search = pyhopper.Search(
            {
                "lr": pyhopper.float(0.01, 0.2, log=True),  # Only search the learning rate
            }
        )

        # Run K-Fold cross-validation using pyhopper
        search.run(
            pyhopper.wrap_n_times(kfold_objective, n=FOLDS, yield_after=0, pass_index_arg=True),
            runtime='150m',  # Number of parameter search steps
            direction='maximize',  # Maximize the validation accuracy
            quiet=False,
        )

        search.save(f"{LOG_DIR}/{NAME}_{dataset_name}.ckpt")
        search_results = {
            'fs': list(search.history.fs),
            'steps': list(search.history.steps),
            'best_fs': list(search.history.best_fs),
            'best_f': search.history.best_f,
            'history': list(search.history),
            'best_params': list(search.history)[list(search.history.fs).index(max(list(search.history.fs)))],
        }

        import json

        with open(f"{LOG_DIR}/{NAME}_{dataset_name}_results.json", "w") as f:
            json.dump(search_results, f, indent=2)


# Example of running the main function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning for ResNet20 on CIFAR10 or CIFAR100")
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100'], required=True, help='Dataset to use')
    parser.add_argument('--name', type=str, required=True, help='Name of the hyperparameter tuning run')
    parser.add_argument('--pruning-percentile', type=int, default=80, help='Pruning percentile for early stopping')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for K-Fold cross-validation')    

    args = parser.parse_args()

    NAME = args.name
    DATASET = args.dataset
    PERCENTILE_PRUNE = args.pruning_percentile
    FOLDS = args.folds

    main(dataset_name=DATASET, requested_device='cuda')  # Change this to "CIFAR100" if needed
