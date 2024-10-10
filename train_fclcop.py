import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from tqdm import tqdm  # Progress bar
from loss_functions.filter_at import FilterAttentionTransfer
from loss_functions.vanilla import VanillaKDLoss
import numpy as np
import json  # For saving metrics to a JSON file

# Assuming ResNet56 and ResNet20 exist and take num_classes as a parameter
from models.resnet import *

if __name__ == '__main__':
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
                    "milestones": [60, 80, 90]
                }
            }
        ],
        "training": {
            "max_epochs": 100,
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
    num_workers = 4
    persistent_workers = True if num_workers > 0 else False
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=hparams['training']['batch_size'], shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True, drop_last=True)

    num_workers = 2
    persistent_workers = True if num_workers > 0 else False
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=hparams['training']['batch_size'], shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)

    # Models
    num_classes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained teacher
    teacher = resnet56(num_classes=num_classes).to(device)
    # teacher = resnet20x2(num_classes=num_classes).to(device)
    teacher_weights_path = r'C:\Users\jonat\OneDrive\UNIV stuff\CS4\COS700\Dev\KernelBasedKD\teacher_models\models\resnet56\resnet56_cifar100_73p18.pth'
    # teacher_weights_path = r'C:\Users\jonat\OneDrive\UNIV stuff\CS4\COS700\Dev\KernelBasedKD\teacher_models\models\resnet20x2\resnet20x2@CIFAR100_params3_#2.pth'
    teacher.load(teacher_weights_path)
    teacher = teacher.to(device)
    teacher.eval()
    teacher.set_hook_device_state('same')

    # Initialize student
    student = resnet20(num_classes=num_classes).to(device)

    teacher_state_dict = teacher.state_dict()
    student_state_dict = student.state_dict()

    # Copy convolutional weights
    # for key in student_state_dict:
    #     if 'conv' in key and key in teacher_state_dict:
    #         teacher_weight = teacher_state_dict[key]
    #         student_weight = student_state_dict[key]
            
    #         # Downsampling by taking every second filter from the teacher model
    #         if teacher_weight.size(0) > student_weight.size(0):
    #             student_state_dict[key] = teacher_weight[:student_weight.size(0), :student_weight.size(1), :, :]
    #         else:
    #             student_state_dict[key] = teacher_weight

    # Load the modified weights into the student model
    # student.load_state_dict(student_state_dict)

    student.fc.weight.data = teacher.fc.weight.data.clone()
    student.fc.bias.data = teacher.fc.bias.data.clone()

    student.fc.weight.requires_grad = False
    student.fc.bias.requires_grad = False

    student.set_hook_device_state('same')

    # Optimizer for student
    optimizer_student = optim.SGD(student.parameters(), **hparams['optimizer']['parameters'])

    # Scheduler for student
    scheduler_student = MultiStepLR(optimizer_student, **hparams['schedulers'][0]['parameters'])

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # KD loss functions
    kd_criterion = FilterAttentionTransfer(student=student, teacher=teacher, map_p=2, loss_p=2, mean_targets=['C_out', 'C_in'], use_abs=False, layer_groups='all')
    vanilla_kd_criterion = VanillaKDLoss(4)
    beta = 10.0

    # Define the evaluate function
    def evaluate(model, dataloader):
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

    # Training loop
    for epoch in range(hparams['training']['max_epochs']):
        teacher.eval()
        student.train()

        running_loss_student = 0.0
        running_kd_loss = 0.0

        if epoch == 70:
            # reenable the fc layer
            student.fc.weight.requires_grad = True
            student.fc.bias.requires_grad = True

        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{hparams['training']['max_epochs']}", ncols=100)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer_student.zero_grad()

            # Forward
            outputs_student = student(inputs)

            # # Obtain teacher outputs without computing gradients
            # with torch.no_grad():
            #     outputs_teacher = teacher(inputs)

            # Loss
            loss_ce = criterion(outputs_student, targets)
            # vanilla_loss = vanilla_kd_criterion(outputs_student, outputs_teacher, targets, features=inputs, indices=None)
            # loss_kd = beta * kd_criterion(outputs_student, outputs_teacher, targets, features=inputs, indices=None)
            loss_student = 1.0 * loss_ce
            # loss_student = 1.0 * loss_ce + 0.0 * vanilla_loss + loss_kd

            # Backward + optimize
            loss_student.backward()
            optimizer_student.step()

            # Track loss
            running_loss_student += loss_student.item()
            # running_kd_loss += loss_kd.item()

            # Print running loss every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"[{batch_idx + 1}] S. L.: {running_loss_student / (batch_idx + 1):.4f}  |  KD L.: {running_kd_loss / (batch_idx + 1):.4f}")

        # Step the scheduler
        scheduler_student.step()

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            student_accuracy = evaluate(student, testloader)
            print(f'Epoch [{epoch + 1}/{hparams["training"]["max_epochs"]}] '
                  f'Student Accuracy: {student_accuracy:.2f}%')

    # Final evaluation
    student_accuracy = evaluate(student, testloader)

    print(f'Final Student Model Accuracy: {student_accuracy:.2f}%')
