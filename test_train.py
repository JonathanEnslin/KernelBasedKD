import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp

from models.resnet import resnet20, resnet110  # Custom ResNet for CIFAR datasets

if __name__ == "__main__":

  # Define transformations for the training and test sets
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

  # Load the CIFAR-100 dataset
  trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
  trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
  testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

  # Initialize the nn model
  model = resnet20()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # Define the loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
  scaler = amp.GradScaler()

  # Define the cosine annealing scheduler
#   scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  # scheduler that divides by 5 at every 45th epoch
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

  # Initialize TensorBoard writer
  writer = SummaryWriter("runs/resnet20_cifar100_params2_smaller_test_batch_size")

  # Training function
  def train(epoch):
      model.train()
      running_loss = 0.0
      for i, (inputs, labels) in enumerate(trainloader):
          inputs, labels = inputs.to(device), labels.to(device)
          
          optimizer.zero_grad()
          
          with amp.autocast():
              outputs = model(inputs)
              loss = criterion(outputs, labels)
          
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
          
          running_loss += loss.item()
          
          if i % 100 == 99:  # Log every 100 mini-batches
              print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.3f}')
              writer.add_scalar('training_loss', running_loss / 100, epoch * len(trainloader) + i)
              running_loss = 0.0
      
      # Step the scheduler
      scheduler.step()

  # Test function
  def test(epoch):
      model.eval()
      correct = 0
      total = 0
      with torch.no_grad():
          for inputs, labels in testloader:
              inputs, labels = inputs.to(device), labels.to(device)
              outputs = model(inputs)
              _, predicted = torch.max(outputs, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
      
      accuracy = 100 * correct / total
      print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')
      writer.add_scalar('test_accuracy', accuracy, epoch)

  # Main training loop
  num_epochs = 200
  for epoch in range(num_epochs):
      train(epoch)
      test(epoch)

  # Close the TensorBoard writer
  writer.close()
