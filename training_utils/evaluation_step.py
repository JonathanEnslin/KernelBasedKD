import torch
import torch.cuda.amp as amp
from sklearn.metrics import f1_score
from utils.log_utils import create_log_entry, log_to_csv

class EvaluationStep:
    def __init__(self, model, dataloader, criterion, device, writer, start_time, autocast, mode='test', early_stopping=None, best_model_tracker=None, logger=print, no_write=False):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.writer = writer
        self.start_time = start_time
        self.autocast = autocast
        self.mode = mode
        self.early_stopping = early_stopping if mode == 'validation' else None
        self.best_model_tracker = best_model_tracker if mode == 'validation' else None
        self.logger = logger
        self.no_write = no_write

    def __call__(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        top5_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels, _ in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                total_samples += labels.size(0)

                with self.autocast(self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_labels.append(labels.cpu())
                all_preds.append(predicted.cpu())

                # Calculate top-5 error
                top5 = outputs.topk(5, dim=1)[1]
                top5_correct += (top5 == labels.view(-1, 1)).sum().item()

        # Concatenate all predictions and labels for metrics calculation
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        # Calculate metrics
        epoch_loss = running_loss / len(self.dataloader)
        accuracy = 100 * (all_preds == all_labels).sum().item() / len(all_labels)
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')
        top5_error = 100 * (1 - top5_correct / total_samples)

        # Log metrics
        prefix = 'validation' if self.mode == 'validation' else 'test'
        self.logger(f'--> {self.mode.capitalize()} accuracy: {accuracy:.2f}%')
        
        if not self.no_write:
            self.writer.add_scalar(f'{prefix}_loss', epoch_loss, epoch)
            self.writer.add_scalar(f'{prefix}_accuracy', accuracy, epoch)
            self.writer.add_scalar(f'{prefix}_f1_score', f1, epoch)
            self.writer.add_scalar(f'{prefix}_top5_error', top5_error, epoch)


            log_entry = create_log_entry(epoch, self.mode, epoch_loss, accuracy, f1, self.start_time, self.device, top5_error)
            self.logger.log_to_csv(log_entry)

            if self.mode == 'validation':
                if self.best_model_tracker:
                    self.best_model_tracker(epoch_loss, self.model)

                if self.early_stopping:
                    self.early_stopping(epoch_loss)
                    if self.early_stopping.early_stop:
                        self.logger("Early stopping triggered")
                        return True

        return False
