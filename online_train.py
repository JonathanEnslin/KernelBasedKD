import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from tqdm import tqdm  # Progress bar
from loss_functions.filter_at import FilterAttentionTransfer
from loss_functions.vanilla import VanillaKDLoss
import argparse  # For command-line arguments
import numpy as np
import json  # For saving metrics to a JSON file

# Assuming ResNet56 and ResNet20 exist and take num_classes as a parameter
from models.resnet import resnet20, resnet56

def save_model_weights(model, file_path):
    # Get all kernel weights, detached from the computation graph
    weights = model.get_all_kernel_weights(detached=True)
    # Convert to numpy arrays
    weights_np = [w.cpu().numpy() for w in weights]
    # Save to an npz file
    np.savez(file_path, *weights_np)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Training script with KD options.')
    parser.add_argument('--save_model_name', type=str, default='trained_model', help='Filename to save the model (without extension)')
    parser.add_argument('--two_way_kd', action='store_true', help='Flag to enable both-way KD (teacher <-> student)')
    args = parser.parse_args()

    # Hyperparameters
    hparams = {
        "optimizer": {
            "type": "SGD",
            "parameters": {
                "lr": 0.1,
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": 0.0005
            }
        },
        "schedulers": [
            {
                "type": "MultiStepLR",
                "parameters": {
                    "gamma": 0.2,
                    "milestones": [60, 80, 90, 100]
                }
            }
        ],
        "training": {
            "max_epochs": 110,
            "batch_size": 128
        }
    }

    # Data loading
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

    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=hparams['training']['batch_size'], shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True, drop_last=True)

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=hparams['training']['batch_size'], shuffle=False, num_workers=2, persistent_workers=True)

    # Models
    num_classes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = resnet20(num_classes=num_classes).to(device)
    teacher = resnet20(num_classes=num_classes).to(device)

    teacher.set_hook_device_state('same')
    student.set_hook_device_state('same')

    # Optimizers
    optimizer_teacher = optim.SGD(teacher.parameters(), **hparams['optimizer']['parameters'])
    optimizer_student = optim.SGD(student.parameters(), **hparams['optimizer']['parameters'])

    # Schedulers
    scheduler_teacher = MultiStepLR(optimizer_teacher, **hparams['schedulers'][0]['parameters'])
    scheduler_student = MultiStepLR(optimizer_student, **hparams['schedulers'][0]['parameters'])

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # KD variables
    kd_criterion = FilterAttentionTransfer(student=student, teacher=teacher, map_p=1.0, loss_p=2, use_abs=False, mean_targets=[], layer_groups='all')
    teacher_kd_criterion = FilterAttentionTransfer(student=teacher, teacher=student, map_p=1.0, loss_p=2, use_abs=False, mean_targets=[], layer_groups='all')
    vanilla_kd_criterion = VanillaKDLoss(4)
    beta = 1000.0

    # Training loop
    for epoch in range(hparams['training']['max_epochs']):
        teacher.train()
        student.train()

        running_loss_teacher = 0.0
        running_loss_student = 0.0
        running_kd_loss = 0.0

        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{hparams['training']['max_epochs']}", ncols=100)

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer_teacher.zero_grad()
            optimizer_student.zero_grad()

            # Forward
            outputs_teacher = teacher(inputs)
            outputs_student = student(inputs)

            # Loss
            loss_ce_teacher = criterion(outputs_teacher, targets)
            loss_ce_student = criterion(outputs_student, targets)

            loss_kd_student = beta * kd_criterion(outputs_student, outputs_teacher, targets, features=inputs, indices=None)
            vanilla_loss_student = vanilla_kd_criterion(outputs_student, outputs_teacher, targets, features=inputs, indices=None)
            loss_student = 0.8 * loss_ce_student + 0.2 * vanilla_loss_student + loss_kd_student

            # Optional two-way KD
            if args.two_way_kd:
                loss_kd_teacher = beta * teacher_kd_criterion(outputs_teacher, outputs_student, targets, features=inputs, indices=None)
                vanilla_teacher_loss = vanilla_kd_criterion(outputs_teacher, outputs_student, targets, features=inputs, indices=None)
                loss_ce_teacher = 0.8 * loss_ce_teacher + 0.2 * vanilla_teacher_loss + loss_kd_teacher
                # Backward for teacher
                loss_ce_teacher.backward()
                optimizer_teacher.step()

            # Backward + optimize for student
            loss_student.backward()
            optimizer_student.step()

            # Track loss
            running_loss_teacher += loss_ce_teacher.item()
            running_loss_student += loss_student.item()
            running_kd_loss += loss_kd_student.item()

            # Print running loss every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"[{batch_idx + 1}] T. L.: {running_loss_teacher / (batch_idx + 1):.4f}  --  "
                      f"S. L.: {running_loss_student / (batch_idx + 1):.4f}  |  KD L.: {running_kd_loss / (batch_idx + 1):.4f}")

        # Step the schedulers
        scheduler_teacher.step()
        scheduler_student.step()

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            teacher_accuracy = evaluate(teacher, testloader, device)
            student_accuracy = evaluate(student, testloader, device)
            print(f'Epoch [{epoch + 1}/{hparams["training"]["max_epochs"]}] '
                  f'Teacher Accuracy: {teacher_accuracy:.2f}% '
                  f'Student Accuracy: {student_accuracy:.2f}%')

    # Final evaluation
    teacher_accuracy = evaluate(teacher, testloader, device)
    student_accuracy = evaluate(student, testloader, device)

    print(f'Final Teacher Model Accuracy: {teacher_accuracy:.2f}%')
    print(f'Final Student Model Accuracy: {student_accuracy:.2f}%')

    # Save model weights at the end of training
    save_model_weights(teacher, rf'e:\DLModels\model_weights\{args.save_model_name}_teacher.npz')
    save_model_weights(student, rf'e:\DLModels\model_weights\{args.save_model_name}_student.npz')
