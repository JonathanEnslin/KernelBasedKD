import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from tqdm import tqdm  # Progress bar
from loss_functions.factor_transfer import FTLoss
from loss_functions.vanilla import VanillaKDLoss
import numpy as np
import json  # For saving metrics to a JSON file

# Assuming ResNet56 and ResNet20 exist and take num_classes as a parameter
from models.resnet import resnet20, resnet56
from models.FT.encoders import Paraphraser, Translator

def save_model_weights(model, title):
    # Get all kernel weights, detached from the computation graph
    weights = model.get_all_kernel_weights(detached=True)
    # Convert to numpy arrays
    weights_np = [w.cpu().numpy() for w in weights]
    # Save to an npz file
    np.savez(rf'e:\DLModels\model_weights\{title}.npz', *weights_np)

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
    teacher_weights_path = r'C:\Users\jonat\OneDrive\UNIV stuff\CS4\COS700\Dev\KernelBasedKD\teacher_models\models\resnet56\resnet56_cifar100_73p18.pth'
    teacher.load(teacher_weights_path)
    teacher = teacher.to(device)
    teacher.eval()
    teacher.set_hook_device_state('same')

    # Run teacher through validation set to get feature map shape
    print("Running teacher model on validation set to get feature map shape")
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            teacher(images)
    print("Teacher model feature map shape:", teacher.get_post_activation_fmaps()[-1].shape)

    # Load paraphraser
    ft_k = 0.5
    ft_use_bn = False
    paraphraser = Paraphraser(teacher.get_post_activation_fmaps()[-1].shape, k=ft_k, use_bn=ft_use_bn).to(device)
    paraphraser_path = r'C:\Users\jonat\OneDrive\UNIV stuff\CS4\COS700\Dev\KernelBasedKD\teacher_models\models\resnet56\paraphraser\resnet56_cifar100_73p18.paraphraser_k_0.5_bn0.pt'
    paraphraser.load(paraphraser_path)
    paraphraser = paraphraser.to(device)
    paraphraser.set_hook_device_state('same')
    paraphraser.eval()

    # Initialize student
    student = resnet20(num_classes=num_classes).to(device)
    student.set_hook_device_state('same')

    # Run student through validation set to get feature map shape
    print("Running student model on validation set to get feature map shape")
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            student(images)
    print("Student model feature map shape:", student.get_post_activation_fmaps()[-1].shape)

    # Initialise the translator
    translator = Translator(student.get_post_activation_fmaps()[-1].shape, teacher.get_post_activation_fmaps()[-1].shape, k=ft_k, use_bn=ft_use_bn).to(device)
    translator.set_hook_device_state('same')
    translator.train()

    # Optimizer for student
    optimizer_student = optim.SGD(student.parameters(), **hparams['optimizer']['parameters'])
    optimizer_factor = optim.SGD(translator.parameters(), lr=hparams['optimizer']['parameters']['lr'], momentum=0.9, weight_decay=5e-4)

    # Scheduler for student
    scheduler_student = MultiStepLR(optimizer_student, **hparams['schedulers'][0]['parameters'])

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # KD loss functions
    kd_criterion = FTLoss(translator, paraphraser, student, teacher, p=1)
    vanilla_kd_criterion = VanillaKDLoss(4)
    beta = 500

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

        mean_dim01 = filter_weights.mean(dim=(0,1))
        abs_mean_dim01 = filter_weights.abs().mean(dim=(0,1))
        
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
            'mean_dim01': mean_dim01.cpu().numpy().tolist(),
            'abs_mean_dim01': abs_mean_dim01.cpu().numpy().tolist(),
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

    # Initialize lists to collect metrics
    student_metrics_collection = []

    # Training loop
    for epoch in range(hparams['training']['max_epochs']):
        teacher.eval()
        # paraphraser.eval()
        student.train()
        # translator.train()

        running_loss_student = 0.0
        running_kd_loss = 0.0

        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{hparams['training']['max_epochs']}", ncols=100)
        # Save the student model weights at the beginning of each epoch
        save_model_weights(student, f'student_epoch_{epoch}')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Track metrics every N minibatches
            N = 25
            if (batch_idx) % N == 0:
                student_metrics = collect_metrics_for_network_layers([w.cpu() for w in student.get_all_kernel_weights(True)])
                
                student_metrics_collection.append({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'metrics': student_metrics
                })

            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer_student.zero_grad()
            optimizer_factor.zero_grad()

            # Forward
            outputs_student = student(inputs)

            # Obtain teacher outputs without computing gradients
            with torch.no_grad():
                outputs_teacher = teacher(inputs)

            # Loss
            loss_ce = criterion(outputs_student, targets)
            vanilla_loss = vanilla_kd_criterion(outputs_student, outputs_teacher, targets, features=inputs, indices=None)
            loss_kd = beta * kd_criterion(outputs_student, outputs_teacher, targets, features=inputs, indices=None)
            loss_student = 0.2 * loss_ce + 0.8 * vanilla_loss +  loss_kd

            # Backward + optimize
            loss_student.backward()
            optimizer_student.step()
            optimizer_factor.step()

            # Track loss
            running_loss_student += loss_student.item()
            running_kd_loss += loss_kd.item()

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

    # Save metrics to JSON files
    with open(r'e:\DLModels\student_metrics.json', 'w') as f:
        json.dump(student_metrics_collection, f, indent=4)

    # Save the final student model
    save_model_weights(student, 'student_final')

    # Final evaluation
    student_accuracy = evaluate(student, testloader)

    print(f'Final Student Model Accuracy: {student_accuracy:.2f}%')
