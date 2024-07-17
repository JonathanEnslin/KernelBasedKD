import torch
import torch.cuda.amp as amp
from sklearn.metrics import f1_score
from utils.log_utils import create_log_entry, log_to_csv

class TrainStep:
    def __init__(self, model, trainloader, criterion, optimizer, scaler, schedulers, device, writer, csv_file, start_time):
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

    def __call__(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        for i, (inputs, labels) in enumerate(self.trainloader):
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            if i % 100 == 99:  # Log every 100 mini-batches
                print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.3f}')
                self.writer.add_scalar('training_loss', running_loss / 100, epoch * len(self.trainloader) + i)
                running_loss = 0.0
        
        # Step the schedulers
        for scheduler in self.schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(running_loss)
            else:
                scheduler.step()

        # Calculate metrics
        epoch_loss = running_loss / len(self.trainloader)
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')
        self.writer.add_scalar('training_epoch_loss', epoch_loss, epoch)
        self.writer.add_scalar('training_accuracy', accuracy, epoch)
        self.writer.add_scalar('training_f1_score', f1, epoch)

        log_entry = create_log_entry(epoch, 'train', epoch_loss, accuracy, f1, self.start_time)
        log_to_csv(self.csv_file, log_entry)
