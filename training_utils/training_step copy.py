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
        running_time = 0.0
        for i, (inputs, labels, indices) in enumerate(self.trainloader):
            loop_start = time.time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
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
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Calculate top-5 error
            # top5 = outputs.topk(5, dim=1)[1]
            # top5_correct += sum([1 if labels[j] in top5[j] else 0 for j in range(len(labels))])
            
            if i % 100 == 0:  # Log every 100 mini-batches
                self.logger(f'[Epoch {epoch+1}, Batch {i+1}/{len(self.trainloader)}] Loss: {running_loss / 100:.3f}')
                self.writer.add_scalar('training_loss', running_loss / 100, epoch * len(self.trainloader) + i)
                running_loss = 0.0
            
            running_time += time.time() - loop_start
        
        self.logger(f"Training duration: {running_time:.2f}")
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
        top5_error = 100 * (1 - top5_correct / len(all_labels))

        self.writer.add_scalar('training_epoch_loss', epoch_loss, epoch)
        self.writer.add_scalar('training_accuracy', accuracy, epoch)
        self.writer.add_scalar('training_f1_score', f1, epoch)
        self.writer.add_scalar('training_top5_error', top5_error, epoch)

        log_entry = create_log_entry(epoch, 'train', epoch_loss, accuracy, f1, self.start_time, self.device, top5_error)
        log_to_csv(self.csv_file, log_entry)
