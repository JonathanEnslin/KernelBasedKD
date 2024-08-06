import torch
import torch.cuda.amp as amp
from sklearn.metrics import f1_score
from utils.log_utils import create_log_entry, log_to_csv
import time

class TrainStep:
    def __init__(self, model, trainloader, criterion, optimizer, scaler, schedulers, device, writer, csv_file, start_time, autocast, is_kd=False, logger=print):
        self.model = model
        self.trainloader = trainloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.schedulers = schedulers
        self.device = device
        self.writer = writer
        self.csv_file = csv_file
        self.start_time = start_time
        self.is_kd = is_kd
        self.autocast = autocast
        self.logger = logger

    def __call__(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        top5_correct = 0
        total_samples = 0  # Keep track of the total number of samples
        for i, (inputs, labels, indices) in enumerate(self.trainloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            total_samples += labels.size(0)
            
            self.optimizer.zero_grad()

            with self.autocast(self.device.type):
                outputs = self.model(inputs)
                if self.is_kd:
                    loss = self.criterion(outputs, labels, features=inputs, indices=indices)
                else:
                    loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())

            # Calculate top-5 error
            top5 = outputs.topk(5, dim=1)[1]
            top5_correct += (top5 == labels.view(-1, 1)).sum().item()
            
            if i % 100 == 0:  # Log every 100 mini-batches
                avg_loss = running_loss / (i + 1)  # Average loss over the batches so far
                self.logger(f'[Epoch {epoch+1}, Batch {i}/{len(self.trainloader) - 1}] Loss: {avg_loss:.3f}')
                self.writer.add_scalar('training_loss', avg_loss, epoch * len(self.trainloader) + i)
                
        # Step the schedulers
        for scheduler in self.schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(running_loss)
            else:
                scheduler.step()

        # Concatenate all predictions and labels for metrics calculation
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        # Calculate metrics
        epoch_loss = running_loss / len(self.trainloader)
        accuracy = 100 * (all_preds == all_labels).sum().item() / len(all_labels)
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')
        top5_error = 100 * (1 - top5_correct / total_samples)

        self.writer.add_scalar('training_epoch_loss', epoch_loss, epoch)
        self.writer.add_scalar('training_accuracy', accuracy, epoch)
        self.writer.add_scalar('training_f1_score', f1, epoch)
        self.writer.add_scalar('training_top5_error', top5_error, epoch)

        log_entry = create_log_entry(epoch, 'train', epoch_loss, accuracy, f1, self.start_time, self.device, top5_error)
        log_to_csv(self.csv_file, log_entry)
