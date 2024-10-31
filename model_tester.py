import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from utils.data.indexed_dataset import IndexedTinyImageNet, IndexedCIFAR10, IndexedCIFAR100
from models.resnet import resnet20, resnet32, resnet56, resnet110

# Define dataset and transformations
dataset = 'TinyImageNet'
# Path to the saved model
model_path = r'C:\Users\jonat\OneDrive\UNIV stuff\CS4\COS700\Dev\KernelBasedKD\teacher_models\models\resnet110\resnet110@TinyImageNet_params10_#2_best.pth'
model_name = 'resnet110'  # Change this according to your model

if __name__ == '__main__':
    num_classes = 100 if dataset == 'CIFAR100' else (10 if dataset == 'CIFAR10' else 200)

    # Mean and std for CIFAR-10 and CIFAR-100
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    tiny_mean = (0.4802, 0.4481, 0.3975)
    tiny_std = (0.2302, 0.2265, 0.2262)


    # Load the test dataset
    print(f"Loading {dataset} test dataset...")
    if dataset == 'CIFAR10':
        model_init_special_kwargs = {}        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std)
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'CIFAR100':
        model_init_special_kwargs = {}
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std)
        ])
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'TinyImageNet':
        model_init_special_kwargs = {
            'conv1stride': 2,
            'conv1ksize': 5,
            'conv1padding': 2
        }
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)
        ])
        test_dataset = IndexedTinyImageNet(root=r"F:\Development\UNIV Dev\CS4\COS700\Datasets\tiny-64", train=False, download=True, transform=transform)

    print(f"Craeting test loader...")
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # Function to load the model
    def load_model(path, model_name):
        if model_name == 'resnet20':
            model = resnet20(num_classes=num_classes, **model_init_special_kwargs)
        elif model_name == 'resnet32':
            model = resnet32(num_classes=num_classes, **model_init_special_kwargs)
        elif model_name == 'resnet56':
            model = resnet56(num_classes=num_classes, **model_init_special_kwargs)
        elif model_name == 'resnet110':
            model = resnet110(num_classes=num_classes, **model_init_special_kwargs)
        else:
            raise ValueError(f"Unknown model name {model_name}")
        
        model.load(path)
        return model

    # Function to evaluate the model
    def evaluate(model, data_loader):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        all_labels = []
        all_preds = []
        total_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            index = 0
            for data in data_loader:
                print(f"Batch {index} / {len(data_loader)}")
                index += 1
                images, labels, _ = data
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
                top1_correct += (preds == labels).sum().item()
                top5_correct += sum([1 if labels[i] in torch.topk(outputs[i], 5).indices else 0 for i in range(len(labels))])
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(data_loader)
        top1_error = 1 - top1_correct / total_samples
        top5_error = 1 - top5_correct / total_samples
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return accuracy, f1, top1_error, top5_error, avg_loss


    print("Loading model...")
    # Load the model
    model = load_model(model_path, model_name)

    print("Evaluating model...")
    # Evaluate the model on the test set
    accuracy, f1, top1_error, top5_error, loss = evaluate(model, test_loader)

    print("==========================")
    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Top-1 Error: {top1_error:.4f}")
    print(f"Top-5 Error: {top5_error:.4f}")
    print(f"Loss: {loss:.4f}")
