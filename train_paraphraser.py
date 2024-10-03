import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import argparse
import os
from models.resnet import resnet56, resnet110
from models.FT.encoders import Paraphraser

def train_paraphraser(paraphraser, teacher_model, dataloader, optimizer, criterion, device):
    paraphraser.train()
    running_loss = 0.0
    for batch_idx, (images, _) in enumerate(dataloader, 1):
        images = images.to(device)
        optimizer.zero_grad()
        feature_maps = teacher_model(images)
        feature_maps = teacher_model.get_post_activation_fmaps(detached=True)[-1].to(device)
        _, rec = paraphraser(feature_maps)
        loss = criterion(rec, feature_maps)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
    return running_loss / len(dataloader)


def validate_paraphraser(paraphraser, teacher_model, dataloader, criterion, device):
    paraphraser.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feature_maps = teacher_model(images)
            feature_maps = teacher_model.get_post_activation_fmaps(detached=True)[-1].to(device)
            _, rec = paraphraser(feature_maps)
            loss = criterion(rec, feature_maps)
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

def main():
    parser = argparse.ArgumentParser(description="Train Paraphraser on pretrained teacher model")
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100'], required=True, help='Dataset to use')
    parser.add_argument('--teacher-path', type=str, required=True, help='Filepath of the pretrained teacher network')
    parser.add_argument('--k', type=float, default=0.5, help='Hyperparameter k for the paraphraser')
    parser.add_argument('--use-bn', action='store_true', help='Use batch normalization')
    args = parser.parse_args()

    # Determine number of classes based on dataset
    num_classes = 10 if args.dataset == "CIFAR10" else 100

    # Determine the output directory
    teacher_dir = os.path.dirname(args.teacher_path)
    output_dir = os.path.join(teacher_dir, 'paraphraser')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset with proper transforms
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

    # Split dataset into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, persistent_workers=True)

    # Load the teacher model based on the filename
    teacher_model = load_teacher_model(args.teacher_path, num_classes, device)
    teacher_model.set_hook_device_state('same')

    # Initialize the paraphraser
    # Run the teacher through the validation set to get the shape of the feature maps and the validation accuracy/loss
    print("Running teacher model on validation set to get feature map shape")
    teacher_model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            teacher_model(images)


    print("Initializing paraphraser")
    t_shape = teacher_model.get_post_activation_fmaps()[-1].shape
    paraphraser = Paraphraser(t_shape, k=args.k, use_bn=args.use_bn).to(device)
    paraphraser.to(device)

    # Training setup
    optimizer = optim.SGD(paraphraser.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        train_loss = train_paraphraser(paraphraser, teacher_model, train_loader, optimizer, criterion, device)
        val_loss = validate_paraphraser(paraphraser, teacher_model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the trained paraphraser
    teacher_name = os.path.basename(args.teacher_path).split('.')[0]
    bn_status = 'bn1' if args.use_bn else 'bn0'
    paraphraser_filename = f"{teacher_name}.paraphraser_k_{args.k}_{bn_status}.pt"
    paraphraser_filepath = os.path.join(output_dir, paraphraser_filename)
    torch.save(paraphraser.state_dict(), paraphraser_filepath)
    print(f"Trained paraphraser saved to {paraphraser_filepath}")

if __name__ == "__main__":
    main()
