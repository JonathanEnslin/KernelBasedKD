import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from tqdm import tqdm  # Progress bar
from loss_functions.filter_at import FilterAttentionTransfer
import numpy as np

# Assuming ResNet56 and ResNet20 exist and take num_classes as a parameter
from models.resnet import resnet20, resnet56
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=hparams['training']['batch_size'], shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True, drop_last=True)

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=hparams['training']['batch_size'], shuffle=False, num_workers=2, persistent_workers=True)

    # Models
    num_classes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = resnet56(num_classes=num_classes).to(device)
    student = resnet20(num_classes=num_classes).to(device)
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

    # kd variables
    kd_criterion = FilterAttentionTransfer(student=student, teacher=teacher, map_p=0.5, loss_p=0.5)
    beta = 1000.0

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

    def calculate_entropy(values, num_bins=100):
        # Flatten the values
        flattened = values.view(-1).detach().cpu().numpy()
        
        # Create a histogram (empirical PDF) of the values
        hist, bin_edges = np.histogram(flattened, bins=num_bins, density=True)
        
        # Normalize the histogram to get probabilities
        p = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(p * np.log(p + 1e-9))  # Adding a small value to avoid log(0)
        
        return entropy

    def compute_filter_metrics(filter_weights, num_bins=100):
        # Flatten the filter to compute vector-based metrics
        flattened = filter_weights.view(-1)
        
        # Compute basic statistics
        mean = filter_weights.mean().item()
        abs_mean = filter_weights.abs().mean().item()

        mean_dim0 = filter_weights.mean(dim=0)
        mean_dim1 = filter_weights.mean(dim=1)
        mean_dim01 = filter_weights.mean(dim=(0,1))
        
        abs_mean_dim0 = filter_weights.abs().mean(dim=0)
        abs_mean_dim1 = filter_weights.abs().mean(dim=1)
        abs_mean_dim01 = filter_weights.abs().mean(dim=(0,1))
        
        # Compute angles
        if flattened.is_complex():
            angles = torch.atan2(flattened.imag, flattened.real)
        else:
            angles = torch.atan2(flattened, torch.ones_like(flattened))
        
        angles_mean = angles.mean().item()
        angles_std = angles.std().item()
        
        # Compute norms
        l2_norm = flattened.norm(2).item()  # Euclidean norm
        max_norm = flattened.norm(float('inf')).item()  # Max norm
        
        # Compute sparsity (percentage of zeros)
        sparsity = (flattened == 0).float().mean().item()

        # Compute overall entropy
        entropy = calculate_entropy(filter_weights, num_bins=num_bins)
        
        # Compute entropy map for each (h, w) position across C_in and C_out
        H, W = filter_weights.shape[2], filter_weights.shape[3]
        entropy_map = np.zeros((H, W))
        
        for h in range(H):
            for w in range(W):
                position_values = filter_weights[:, :, h, w]
                entropy_map[h, w] = calculate_entropy(position_values, num_bins=num_bins)
        
        # Parameterized dict
        metrics = {
            'mean': mean,
            'abs_mean': abs_mean,
            'mean_dim0': mean_dim0.numpy().tolist(),
            'mean_dim1': mean_dim1.numpy().tolist(),
            'mean_dim01': mean_dim01.numpy().tolist(),
            'abs_mean_dim0': abs_mean_dim0.numpy().tolist(),
            'abs_mean_dim1': abs_mean_dim1.numpy().tolist(),
            'abs_mean_dim01': abs_mean_dim01.numpy().tolist(),
            'angles_mean': angles_mean,
            'angles_std': angles_std,
            'l2_norm': l2_norm,
            'max_norm': max_norm,
            'sparsity': sparsity,
            'entropy': entropy,
            'entropy_map': entropy_map.tolist()  # Entropy at each (h, w) position
        }
        
        return metrics

    def collect_metrics_for_network_layers(layers_weights):
        metrics_list = []
        for layer_index, layer_weights in enumerate(layers_weights):
            if len(layer_weights.shape) == 4:  # Assuming weights have the shape [C_out, C_in, H, W]
                metrics = compute_filter_metrics(layer_weights)
                metrics_list.append({
                    'layer_index': layer_index,
                    'metrics': metrics
                })
            else:
                print(f"Skipping layer {layer_index} as it does not have a 4D shape.")
        
        return metrics_list



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
            loss_teacher = criterion(outputs_teacher, targets)
            loss_ce = criterion(outputs_student, targets)
            loss_kd = beta * kd_criterion(outputs_student, outputs_teacher, targets, features=inputs, indices=None)
            loss_student = loss_ce + loss_kd

            # Backward + optimize
            loss_teacher.backward()
            optimizer_teacher.step()

            loss_student.backward()
            optimizer_student.step()

            # Track loss
            running_loss_teacher += loss_teacher.item()
            running_loss_student += loss_student.item()
            running_kd_loss += loss_kd.item()

            # Print running loss every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"[{batch_idx + 1}] T. L.: {running_loss_teacher / (batch_idx + 1):.4f}  --  "
                    f"S. L.: {running_loss_student / (batch_idx + 1):.4f}  |  KD L.: {running_kd_loss / (batch_idx + 1):.4f}")

        # Step the schedulers
        scheduler_teacher.step()
        scheduler_student.step()

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            teacher_accuracy = evaluate(teacher, testloader)
            student_accuracy = evaluate(student, testloader)
            print(f'Epoch [{epoch + 1}/{hparams["training"]["max_epochs"]}] '
                f'Teacher Accuracy: {teacher_accuracy:.2f}% '
                f'Student Accuracy: {student_accuracy:.2f}%')

    # Final evaluation
    teacher_accuracy = evaluate(teacher, testloader)
    student_accuracy = evaluate(student, testloader)

    print(f'Final Teacher Model Accuracy: {teacher_accuracy:.2f}%')
    print(f'Final Student Model Accuracy: {student_accuracy:.2f}%')
