import torch
import torch.cuda.amp as amp
from sklearn.metrics import f1_score
from utils.log_utils import create_log_entry
from models.base_model import BaseModel
from loss_functions.vanilla import VanillaKDLoss
import time

class LossHandler:
    def __init__(self, gamma, alpha, beta, student_criterion, teacher_model: BaseModel=None, vanilla_criterion: VanillaKDLoss=None, kd_criterion=None, run_teacher=True):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.student_criterion = student_criterion
        self.teacher_model = teacher_model
        self.vanilla_criterion = vanilla_criterion
        self.kd_criterion = kd_criterion
        self.run_teacher = run_teacher

class DummyCriterion(torch.nn.Module):
    def __init__(self):
        super(DummyCriterion, self).__init__()

    def forward(self, outputs, labels, *args, **kwargs):
        return 0
    

class TrainStep:
    def __init__(self, model, trainloader, optimizer, scaler, schedulers, device, writer, start_time, autocast, loss_handler=None, logger=print):
        self.loss_handler = loss_handler
        if self.loss_handler is None:
            self.is_kd = False
            self.loss_handler = LossHandler(1, 0, 0, torch.nn.CrossEntropyLoss(), DummyCriterion(), None, DummyCriterion(), False)
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.scaler = scaler
        self.schedulers = schedulers
        self.device = device
        self.writer = writer
        self.start_time = start_time
        self.autocast = autocast
        self.logger = logger

        if self.loss_handler.run_teacher and self.loss_handler.teacher_model is None:
            logger("Warning: teacher is None, but is specified to be run", col='yellow')

    def __call__(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        top5_correct = 0
        total_samples = 0  # Keep track of the total number of samples
        running_time1 = 0.0
        running_time2 = 0.0
        running_time3 = 0.0
        for i, (inputs, labels, indices) in enumerate(self.trainloader):
            start_time1 = time.time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            total_samples += labels.size(0)
            
            self.optimizer.zero_grad()

            teacher_logits = None
            if self.loss_handler.run_teacher:
                teacher_logits = self.loss_handler.teacher_model.generate_logits(inputs)

            with self.autocast(self.device.type):
                outputs = self.model(inputs)
                student_loss = self.loss_handler.gamma * self.loss_handler.student_criterion(outputs, labels)
                                
                vanilla_loss = 0
                kd_loss = 0
                if self.loss_handler.alpha != 0 and self.loss_handler.vanilla_criterion is not None:
                    vanilla_loss = self.loss_handler.alpha * self.loss_handler.vanilla_criterion(outputs, labels, teacher_logits, features=inputs, indices=indices)
                if self.loss_handler.beta != 0 and self.loss_handler.kd_criterion is not None:
                    kd_loss = self.loss_handler.beta * self.loss_handler.kd_criterion(outputs, labels, teacher_logits, features=inputs, indices=indices)

                loss = student_loss + vanilla_loss + kd_loss
                
            running_time1 += time.time() - start_time1
            start_time2 = time.time()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_time2 += time.time() - start_time2
            
            start_time3 = time.time()
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
                
            running_time3 += time.time() - start_time3
            # if i % 100 == 0:
            #     print(i, "======================================================")
            #     print(f"Time taken for forward pass: {running_time1:.2f}s")
            #     print(f"Time taken for backward pass: {running_time2:.2f}s")
            #     print(f"Time taken for metric calculations: {running_time3:.2f}s")
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
        self.logger.log_to_csv(log_entry)
