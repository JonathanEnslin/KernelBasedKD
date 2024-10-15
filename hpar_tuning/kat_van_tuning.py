import os
import sys
import time
import multiprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functools import partial

import pyhopper
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from sklearn.model_selection import KFold
import numpy as np

from utils.data.dataset_getters import get_dataset_info
from utils.logger import Logger
from utils.teacher.teacher_model_handler import TeacherModelHandler
from utils.model_utils import initialize_model
from loss_functions.vanilla import VanillaKDLoss
from loss_functions.attention_transfer import ATLoss
from loss_functions.filter_at import FilterAttentionTransfer

from mock_args import Args as MockArgs
from pruner import prune_if_underperforming


GENERATE_LOGITS = True
GENERATE_FEATURE_MAPS = False

# Fixed parameters
optimizer_params = {
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 0.0005
}

scheduler_params = {
    "gamma": 0.2,
    "milestones": [45, 65, 72]
}

training_params = {
    "max_epochs": 75,
    "batch_size": 128
}

# Function to split the dataset into folds
def prepare_kfold_splits(dataset, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=112)  # Added a fixed random_state for reproducibility
    return list(kf.split(dataset))  # Returns a list of (train_indices, val_indices) tuples

def manual_wrap_n_times(func, n, yield_after=0, pass_index_arg=False):
    def wrapped_func(*args, **kwargs):
        collected_vals = []
        for i in range(n):
            try:
                obj_value = func(*args, **kwargs, eval_index=i if pass_index_arg else None)
                collected_vals.append(obj_value)
            except pyhopper.PruneEvaluation as ppe:
                if len(collected_vals) == 0:
                    raise ppe
                break
        return sum(collected_vals) / len(collected_vals)
    return wrapped_func

# Noisy objective function for K-Folds (Only takes param and eval_index)
def kfold_objective(param, eval_index, folds, full_dataset, num_classes, device, log_dir, 
                    dataset, student_type, logger_tag, percentile_prune1, num_workers, 
                    persistent_workers, shared_accuracies, teacher_type, tea_model, 
                    tea_logits, tea_preactivation_fmaps, tea_postactivation_fmaps, percentile_prune2=40):
    import time

    args = MockArgs()
    args.dataset = dataset
    args.model_name = student_type
    args.param_set = 'tune'

    logger = Logger(args=args, log_to_file=True, data_dir=log_dir, run_tag=logger_tag, teacher_type=teacher_type, kd_set=None)

    logger(f"Starting evaluation for Fold {eval_index + 1}")
    logger(f"Parameters:\n{param}")

    # Get the appropriate fold
    train_indices, val_indices = folds[eval_index]
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # Prepare data loaders
    train_loader = DataLoader(train_subset, batch_size=training_params["batch_size"], shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_subset, batch_size=training_params["batch_size"], shuffle=False)

    # Initialize the model and move it to the device
    start_time = time.time()
    stu_model = initialize_model(student_type, num_classes, device, logger=logger)
    stu_model.to(device)
    stu_model.set_hook_device_state('same')

    # Define loss and optimizer with the fixed parameters and searched lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        stu_model.parameters(),
        lr=param['lr'],  # Use the learning rate from pyhopper search
        momentum=optimizer_params["momentum"],
        nesterov=optimizer_params["nesterov"],
        weight_decay=optimizer_params["weight_decay"]
    )

    # ========================== SETUP OF KD CRITERION ==========================
    # send logits to device
    if tea_logits is not None:
        if isinstance(tea_logits, list):
            tea_logits = [logits.to(device) for logits in tea_logits]
        else:
            tea_logits = tea_logits.to(device)
    vanilla_criterion = VanillaKDLoss(temperature=param['vanilla_temperature'], cached_teacher_logits=tea_logits)
    kd_criterion = FilterAttentionTransfer(
        student=stu_model,
        teacher=tea_model,
        map_p=param["map_p"],
        loss_p=2.0,
        mean_targets=param['mean_targets'],
        use_abs=param['use_abs'],
        layer_groups='final'
    )
    alpha = param['alpha']
    gamma = 1.0 - alpha
    beta = param['beta']

    # Scheduler (MultiStepLR)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=scheduler_params.get("milestones", []),
        gamma=scheduler_params["gamma"]
    )

    # Training loop
    for epoch in range(training_params["max_epochs"]):
        logger(f"Fold {eval_index + 1}, Epoch {epoch + 1}")
        stu_model.train()
        tea_model.eval()
        tracked_train_loss = 0.0
        tracked_vanilla_loss = 0.0
        tracked_kd_loss = 0.0
        tracked_tot_loss = 0.0
        for batch_idx, (inputs, targets, indices) in enumerate(train_loader):
            if batch_idx % 150 == 0:
                logger(f"Batch {batch_idx + 1}/{len(train_loader)}")

            # Move inputs and targets to the device
            inputs, targets = inputs.to(device), targets.to(device)
            tea_logits = None
            if kd_criterion.run_teacher() or vanilla_criterion.run_teacher():
                with torch.no_grad():
                    tea_model.eval()
                    tea_logits = tea_model(inputs)

            optimizer.zero_grad()
            outputs = stu_model(inputs)
            loss = criterion(outputs, targets)
            tracked_train_loss += loss.item()
            vanilla_loss = vanilla_criterion(student_logits=outputs, teacher_logits=tea_logits, labels=targets, features=inputs, indices=indices)
            kd_loss = kd_criterion(student_logits=outputs, teacher_logits=tea_logits, labels=targets, features=inputs, indices=indices)
            tracked_kd_loss += kd_loss.item()
            tracked_vanilla_loss += vanilla_loss.item()
            loss = gamma * loss + alpha * vanilla_loss + beta * kd_loss
            tracked_tot_loss += loss.item()
            loss.backward()
            optimizer.step()

        logger.log_to_csv({'phase': 'train', 
                           'epoch': epoch, 
                           'student_loss': tracked_train_loss / len(train_loader), 
                           'accuracy': -1.0, 
                           'vanilla_loss': tracked_vanilla_loss / len(train_loader), 
                           'kd_loss': tracked_kd_loss / len(train_loader),
                           'total_loss': tracked_tot_loss / len(train_loader),
                           'time': time.time(),
                           })
        # Step the scheduler after each epoch
        scheduler.step()

    # Validation step
    stu_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, indices in val_loader:
            # Move inputs and targets to the device
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = stu_model(inputs)
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

    prune_if_underperforming(shared_accuracies, eval_index, folds, logger, percentile_prune1, percentile_prune2)

    end_time = time.time()
    logger(f"Fold {eval_index + 1}, Time taken: {end_time - start_time:.2f} seconds")
    return val_accuracy


def configure_teacher(teacher_type, num_classes, device, logger, teacher_path, trainloader, train_dataset, use_cached_values=False):
    teacher_model = initialize_model(teacher_type, num_classes, device, logger=logger)
    teacher_model_handler = TeacherModelHandler(
        teacher_model=teacher_model,
        teacher_type=teacher_type,
        teacher_file_name=teacher_path,
        device=device,
        num_classes=num_classes,
        printer=logger,
    )

    teacher_model = teacher_model_handler.load_teacher_model()
    teacher_model.to(device)
    teacher_model.set_hook_device_state('same')
    if teacher_model is None:
        exit(1)

    logger("Teacher model loaded, generating logits and feature maps if specified")
    teacher_logits = None
    teacher_layer_groups_preactivation_fmaps = None
    teacher_layer_groups_post_activation_fmaps = None
    teacher_logits, teacher_layer_groups_preactivation_fmaps, teacher_layer_groups_post_activation_fmaps = \
        teacher_model_handler.generate_and_save_teacher_logits_and_feature_maps(
            trainloader, train_dataset, 
            generate_logits=GENERATE_LOGITS and use_cached_values, 
            generate_feature_maps=GENERATE_FEATURE_MAPS and use_cached_values
        ) 
    
    del teacher_model_handler
    if not use_cached_values:
        teacher_logits = None
        teacher_layer_groups_preactivation_fmaps = None
        teacher_layer_groups_post_activation_fmaps = None
        logger("Teacher model loaded")
    else:
        logger("Teacher model loaded and feature maps generated")

    return teacher_model, teacher_logits, teacher_layer_groups_preactivation_fmaps, teacher_layer_groups_post_activation_fmaps


# Main section to load dataset and perform K-fold splits once
def main(dataset_name="CIFAR10", requested_device="cuda", runtime='xxx', log_dir="hpar_tuning_logs/vanilla", 
         name='no_name_provided', folds_num=5, pruning_percentile1=80, pruning_percentile2=40, num_workers=0, persistent_workers=False, 
         student_type='resnet20', teacher_type='resnet56', teacher_path='xxx', use_cached_values=False, disable_vanilla=False):
    # Set the device based on the availability of CUDA
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset info
    dataset_class, num_classes, transform_train, transform_test = get_dataset_info(dataset_name)

    # Load the dataset (indexed dataset with transforms)
    full_dataset = dataset_class(root='./data', train=True, download=True, transform=transform_train)

    # Get a train loader for the full dataset
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=0, persistent_workers=False)
    teacher_model, teacher_logits, teacher_layer_groups_preactivation_fmaps, teacher_layer_groups_post_activation_fmaps =\
          configure_teacher(
                teacher_type=teacher_type, 
                num_classes=num_classes,
                device=device, 
                logger=print, 
                teacher_path=teacher_path, 
                trainloader=full_loader, 
                train_dataset=full_dataset,
                use_cached_values=use_cached_values
            )

    # Perform K-Fold splitting once
    folds = prepare_kfold_splits(full_dataset, n_splits=folds_num)

    # Initialize a multiprocessing-safe list for storing accuracies
    with multiprocessing.Manager() as manager:
        shared_accuracies = manager.list()

        search: pyhopper.Search = pyhopper.Search(
            {
                "lr": 0.11,
                # "lr": pyhopper.float(0.01, 0.2, log=True, init=0.1),  # Only search the learning rate
                "vanilla_temperature": pyhopper.float(4.0, 10.0, log=False, init=5.0) if not disable_vanilla else 10.0,
                "alpha": pyhopper.float(0.4, 1.0, init=0.8) if not disable_vanilla else 0.0,
                "beta": pyhopper.float(0.0, 8000.0),
                "mean_targets": pyhopper.choice([['C_out', 'C_in'], ['C_out'], ['C_in'], []], init_index=0),
                "use_abs": pyhopper.choice([True, False], init_index=0),
                "map_p": pyhopper.float(0.5, 4.0, init=2.0),
            }
        )

        # Run K-Fold cross-validation using pyhopper
        search.run(
            manual_wrap_n_times(partial(
                kfold_objective,
                folds=folds,
                full_dataset=full_dataset,
                num_classes=num_classes,
                device=device,
                log_dir=log_dir,
                dataset=dataset_name,
                student_type=student_type,
                logger_tag=name,
                percentile_prune1=pruning_percentile1,
                percentile_prune2=pruning_percentile2,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                shared_accuracies=shared_accuracies,
                teacher_type=teacher_type,
                tea_model=teacher_model,
                tea_logits=teacher_logits,
                tea_preactivation_fmaps=teacher_layer_groups_preactivation_fmaps,
                tea_postactivation_fmaps=teacher_layer_groups_post_activation_fmaps
            ), n=folds_num, yield_after=0, pass_index_arg=True),
            runtime=runtime,  # Number of parameter search steps
            direction='maximize',  # Maximize the validation accuracy
            quiet=False,
        )

        print(f"Best: {search.best_f}")
        print(search.best)

        search.save(f"{log_dir}/{name}_{dataset_name}.ckpt")
        search_results = {
            'fs': list(search.history.fs),
            'steps': list(search.history.steps),
            'best_fs': list(search.history.best_fs),
            'best_f': search.history.best_f,
            'history': list(search.history),
            'best_params': list(search.history)[list(search.history.fs).index(max(list(search.history.fs)))],
        }

        import json

        with open(f"{log_dir}/{name}_{dataset_name}_results.json", "w") as f:
            json.dump(search_results, f, indent=2)

# Example of running the main function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning for ResNet20 on CIFAR10 or CIFAR100")
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100'], required=True, help='Dataset to use')
    parser.add_argument('--name', type=str, required=True, help='Name of the hyperparameter tuning run')
    parser.add_argument('--pruning-percentile', type=int, default=80, help='Pruning percentile for early stopping')
    parser.add_argument('--pruning-percentile2', type=int, default=40, help='Pruning percentile for early stopping')
    parser.add_argument('--runtime', type=str, required=True, help='Runtime for the hyperparameter tuning run')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for the DataLoader')
    parser.add_argument('--persistent-workers', action='store_true', help='Use persistent workers for DataLoader')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for K-Fold cross-validation')
    parser.add_argument('--use_cached_values', action='store_true', help='Use cached logits and feature maps')
    parser.add_argument('--disable-vanilla', action='store_true', help='Disable vanilla KD loss')

    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        requested_device=args.device,
        runtime=args.runtime,
        log_dir="hpar_tuning_logs/kattention",
        name=args.name,
        folds_num=args.folds,
        pruning_percentile1=args.pruning_percentile,
        pruning_percentile2=args.pruning_percentile2,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        student_type='resnet20',
        teacher_type='resnet56',
        teacher_path='resnet56_cifar100_73p18.pth',
        use_cached_values=args.use_cached_values,
        disable_vanilla=args.disable_vanilla,
    )