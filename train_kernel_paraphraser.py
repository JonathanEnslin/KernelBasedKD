import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import argparse
import os

from models.resnet import resnet56, resnet110, ResNet
from models.FT.encoders import Paraphraser
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_paraphraser(paraphraser, teacher_model, dataloader, optimizer, criterion, device, layer_name, array_index):
    paraphraser.train()
    running_loss = 0.0
    data_len = len(dataloader)
    
    for batch_idx in range(data_len):
        optimizer.zero_grad()
        kernel_weights = teacher_model.get_kernel_weights_subset([getattr(teacher_model, f'{layer_name}indices')[array_index]], detached=True)[-1].to(device)
        _, rec = paraphraser(kernel_weights)
        loss = criterion(rec, kernel_weights)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
    return running_loss / len(dataloader)

def validate_paraphraser(paraphraser, teacher_model, dataloader, criterion, device, layer_name, array_index):
    paraphraser.eval()
    val_loss = 0.0
    with torch.no_grad():
        for _ in range(len(dataloader)):
            kernel_weights = teacher_model.get_kernel_weights_subset([getattr(teacher_model, f'{layer_name}indices')[array_index]], detached=True)[-1].to(device)
            _, rec = paraphraser(kernel_weights)
            loss = criterion(rec, kernel_weights)
            val_loss += loss.item()
    return val_loss / len(dataloader)

def load_teacher_model(teacher_path, num_classes, device):
    if 'resnet56' in teacher_path:
        model = resnet56(num_classes=num_classes)
    elif 'resnet110' in teacher_path:
        model = resnet110(num_classes=num_classes)
    else:
        raise ValueError("Teacher model must be either resnet56 or resnet110")
    
    model.load(teacher_path, device=device)
    model.to(device)
    model.eval()
    return model

def decode_paraphraser_filename(filename):
    """ Decode the filename and return relevant information as a dictionary. """
    base = os.path.basename(filename).split('.')[0]
    parts = base.split('_')
    
    teacher_name = parts[0]
    layer_name = parts[2]
    array_index = int(parts[4])
    k_value = float(parts[6])
    bn_status = 'with BN' if parts[7] == 'bn1' else 'without BN'
    
    return {
        'teacher_model': teacher_name,
        'layer_name': layer_name,
        'array_index': array_index,
        'k_value': k_value,
        'batch_norm': bn_status
    }

def main():
    parser = argparse.ArgumentParser(description="Train Paraphraser on pretrained teacher model")
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100'], required=True, help='Dataset to use')
    parser.add_argument('--teacher-path', type=str, required=True, help='Filepath of the pretrained teacher network')
    parser.add_argument('--k', type=float, default=0.5, help='Hyperparameter k for the paraphraser')
    parser.add_argument('--use-bn', action='store_true', help='Use batch normalization')
    parser.add_argument('--layer-name', type=str, choices=['conv1', 'group1', 'group2', 'group3'], required=True, help='Which layer to use')
    parser.add_argument('--array-index', type=int, required=True, help='Array index to select from the chosen layer')
    args = parser.parse_args()

    num_classes = 10 if args.dataset == "CIFAR10" else 100
    teacher_dir = os.path.dirname(args.teacher_path)
    output_dir = os.path.join(teacher_dir, 'paraphraser')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset.upper() == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, persistent_workers=True)

    teacher_model: ResNet = load_teacher_model(args.teacher_path, num_classes, device)
    teacher_model.set_hook_device_state('same')

    t_shape = teacher_model.get_kernel_weights_subset([getattr(teacher_model, f'{args.layer_name}indices')[args.array_index]])[-1].shape
    paraphraser = Paraphraser(t_shape, k=args.k, use_bn=args.use_bn, use_linear_last_activation=True).to(device)

    optimizer = optim.SGD(paraphraser.parameters(), lr=0.8, momentum=0.99, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=45, eta_min=0.0001)
    criterion = nn.L1Loss()
    val_criterion = nn.L1Loss()

    num_epochs = 45
    for epoch in range(num_epochs):
        train_loss = train_paraphraser(paraphraser, teacher_model, train_loader, optimizer, criterion, device, args.layer_name, args.array_index)
        val_loss = validate_paraphraser(paraphraser, teacher_model, val_loader, val_criterion, device, args.layer_name, args.array_index)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.6f}, Learning Rate: {current_lr:.6f}")

    teacher_name = os.path.basename(args.teacher_path).split('.')[0]
    bn_status = 'bn1' if args.use_bn else 'bn0'
    paraphraser_filename = f"{teacher_name}_layer_{args.layer_name}_idx_{args.array_index}_k_{args.k}_{bn_status}.pt"
    paraphraser_filepath = os.path.join(output_dir, paraphraser_filename)
    torch.save(paraphraser.state_dict(), paraphraser_filepath)
    print(f"Trained paraphraser saved to {paraphraser_filepath}")

if __name__ == "__main__":
    main()
