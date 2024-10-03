import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from tqdm import tqdm
from loss_functions.factor_transfer import FTLoss
from loss_functions.filter_factor import KFTLoss
from loss_functions.vanilla import VanillaKDLoss
import numpy as np
import json

from models.resnet import resnet20, resnet56
from models.FT.encoders import Paraphraser, Translator


def save_model_weights(model, title):
    weights = model.get_all_kernel_weights(detached=True)
    weights_np = [w.cpu().numpy() for w in weights]
    np.savez(rf'e:\DLModels\kft\model_weights\{title}.npz', *weights_np)


import os
import re

def decode_paraphraser_filename(filepath):
    """ Decode the filename from the full file path and return relevant information as a dictionary using regex. """
    
    # Extract just the file name from the full path
    filename = os.path.basename(filepath)
    
    # Remove the file extension (e.g., '.pt')
    base_filename = os.path.splitext(filename)[0]
    
    # Define the regex pattern to match the expected filename format
    pattern = r'(?P<teacher_model>resnet\d+)_cifar(?P<dataset>\d+)_\d+p\d+_layer_(?P<layer_name>group\d|conv1)_idx_(?P<array_index>-?\d+)_k_(?P<k_value>\d+\.?\d*)_(?P<bn_status>bn\d)'
    
    # Match the pattern
    match = re.match(pattern, base_filename)
    
    if not match:
        raise ValueError(f"Filename format does not match the expected pattern: {filepath}")
    
    # Extract the matched groups
    teacher_model = match.group('teacher_model')
    dataset = match.group('dataset')
    layer_name = match.group('layer_name')
    array_index = int(match.group('array_index'))
    k_value = float(match.group('k_value'))
    bn_status = 'with BN' if match.group('bn_status') == 'bn1' else 'without BN'
    
    return {
        'teacher_model': teacher_model,
        'dataset': dataset,
        'layer_name': layer_name,
        'array_index': array_index,
        'k_value': k_value,
        'batch_norm': bn_status
    }



def load_paraphraser_translator(file_list, teacher, student, device):
    paraphrasers = []
    translators = []
    optimizers_factors = []
    loss_functions = []

    for file_path in file_list:
        # Decode the paraphraser filename to get layer and array index info
        file_info = decode_paraphraser_filename(file_path)
        layer_name = file_info['layer_name']
        array_index = file_info['array_index']
        k_value = file_info['k_value']
        use_bn = file_info['batch_norm'] == 'with BN'
        
        # Get the shape based on the layer and array index
        teacher_kernel_weights = teacher.get_kernel_weights_subset([getattr(teacher, f'{layer_name}indices')[array_index]])[-1].shape
        student_kernel_weights = student.get_kernel_weights_subset([getattr(student, f'{layer_name}indices')[array_index]])[-1].shape

        # Initialize the paraphraser and translator
        paraphraser = Paraphraser(teacher_kernel_weights, k=k_value, use_bn=use_bn, use_linear_last_activation=True).to(device)
        paraphraser.load(file_path)
        paraphraser.set_hook_device_state('same')
        paraphraser.eval()

        translator = Translator(student_kernel_weights, teacher_kernel_weights, k=k_value, use_bn=use_bn).to(device)
        translator.set_hook_device_state('same')

        # Optimizer for the translator
        optimizer_factor = optim.SGD(translator.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        # Loss function for KD task
        loss_function = KFTLoss(translator, paraphraser, student, teacher, group=layer_name, p=1)

        paraphrasers.append(paraphraser)
        translators.append(translator)
        optimizers_factors.append(optimizer_factor)
        loss_functions.append(loss_function)

    return paraphrasers, translators, optimizers_factors, loss_functions


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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=hparams['training']['batch_size'], shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True, drop_last=True)
    
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=hparams['training']['batch_size'], shuffle=False, num_workers=2, persistent_workers=True)

    # Models
    num_classes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained teacher
    teacher = resnet56(num_classes=num_classes).to(device)
    teacher_weights_path = r'C:\Users\jonat\OneDrive\UNIV stuff\CS4\COS700\Dev\KernelBasedKD\teacher_models\models\resnet56\resnet56_cifar100_73p18.pth'
    teacher.load(teacher_weights_path)
    teacher = teacher.to(device)
    teacher.eval()
    teacher.set_hook_device_state('same')

    # Initialize student
    student = resnet20(num_classes=num_classes).to(device)
    student.set_hook_device_state('same')

    # File list for paraphrasers and translators
    paraphraser_files = [
        r'teacher_models\models\resnet56\paraphraser\resnet56_cifar100_73p18_layer_group1_idx_-1_k_0.5_bn0.pt',
        r'teacher_models\models\resnet56\paraphraser\resnet56_cifar100_73p18_layer_group2_idx_-1_k_0.5_bn0.pt',
        r'teacher_models\models\resnet56\paraphraser\resnet56_cifar100_73p18_layer_group3_idx_-1_k_0.5_bn0.pt',
        # Add more files as needed
    ]
    
    # Load paraphrasers, translators, and their optimizers and loss functions
    paraphrasers, translators, optimizers_factors, loss_functions = load_paraphraser_translator(paraphraser_files, teacher, student, device)
    
    # Optimizer for student
    optimizer_student = optim.SGD(student.parameters(), **hparams['optimizer']['parameters'])
    scheduler_student = MultiStepLR(optimizer_student, **hparams['schedulers'][0]['parameters'])

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # KD loss functions
    vanilla_kd_criterion = VanillaKDLoss(4)
    beta = 500

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

    student_metrics_collection = []

    for epoch in range(hparams['training']['max_epochs']):
        teacher.eval()
        student.train()

        running_loss_student = 0.0
        running_kd_loss = 0.0

        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{hparams['training']['max_epochs']}", ncols=100)

        save_model_weights(student, f'student_epoch_{epoch}')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients for student and each paraphraser-translator pair
            optimizer_student.zero_grad()
            for optimizer_factor in optimizers_factors:
                optimizer_factor.zero_grad()

            # Forward pass for student
            outputs_student = student(inputs)

            # Obtain teacher outputs without computing gradients
            with torch.no_grad():
                outputs_teacher = teacher(inputs)

            # Loss calculation
            loss_ce = criterion(outputs_student, targets)
            vanilla_loss = vanilla_kd_criterion(outputs_student, outputs_teacher, targets, features=inputs, indices=None)
            
            # Calculate KD loss for each paraphraser-translator pair
            total_kd_loss = 0.0
            for loss_fn in loss_functions:
                total_kd_loss += beta * loss_fn(outputs_student, outputs_teacher, targets, features=inputs, indices=None)

            # Final student loss
            loss_student = 1.0 * loss_ce + 0.0 * vanilla_loss + total_kd_loss

            # Backward pass
            loss_student.backward()

            # Step the optimizers for student and all paraphrasers/translators
            optimizer_student.step()
            for optimizer_factor in optimizers_factors:
                optimizer_factor.step()

            # Track running loss
            running_loss_student += loss_student.item()
            running_kd_loss += total_kd_loss.item()

            # Print running loss every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"[{batch_idx + 1}] Student Loss: {running_loss_student / (batch_idx + 1):.4f}  |  KD Loss: {running_kd_loss / (batch_idx + 1):.4f}")

        # Step the student learning rate scheduler
        scheduler_student.step()

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            student_accuracy = evaluate(student, testloader)
            print(f'Epoch [{epoch + 1}/{hparams["training"]["max_epochs"]}] '
                  f'Student Accuracy: {student_accuracy:.2f}%')

    # Save metrics to JSON file
    with open(r'e:\DLModels\kft\student_metrics.json', 'w') as f:
        json.dump(student_metrics_collection, f, indent=4)

    # Save the final student model
    save_model_weights(student, 'student_final')

    # Final evaluation of the student
    student_accuracy = evaluate(student, testloader)
    print(f'Final Student Model Accuracy: {student_accuracy:.2f}%')
