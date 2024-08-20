import torch
import torch.cuda.amp as amp
from sklearn.metrics import f1_score
from utils.log_utils import create_log_entry
from models.base_model import BaseModel
from loss_functions.vanilla import VanillaKDLoss
import time

class LossHandler:
    def __init__(self, gamma, alpha, beta, student_criterion, teacher_model: BaseModel=None, vanilla_criterion: VanillaKDLoss=None, kd_criterion=None):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.student_criterion = student_criterion
        self.teacher_model = teacher_model
        self.vanilla_criterion = vanilla_criterion
        self.kd_criterion = kd_criterion
        self.epoch_step_fn = None
        self.batch_step_fn = None
        self.extern_vars = {} # Can be used to store any external variables, such as ones used for the step functions
        self.eval_criterions = []


    def add_eval_criterion(self, name, criterion):
        self.eval_criterions.append((name, criterion))

    def run_teacher(self):
        vanilla_run = self.vanilla_criterion is not None and self.vanilla_criterion.run_teacher()
        kd_run = self.kd_criterion is not None and self.kd_criterion.run_teacher()
        eval_run = any(crit.run_teacher() for (_, crit) in self.eval_criterions)
        return vanilla_run or kd_run or eval_run

    def set_epoch_step_fn(self, fn):
        self.epoch_step_fn = fn

    def set_batch_step_fn(self, fn):
        self.batch_step_fn = fn

    def epoch_step(self, epoch_idx):
        if self.epoch_step_fn is not None:
            self.epoch_step_fn(epoch_idx)

    def batch_step(self, batch_idx):
        if self.batch_step_fn is not None:
            self.batch_step_fn(batch_idx)
        

class DummyCriterion(torch.nn.Module):
    def __init__(self):
        super(DummyCriterion, self).__init__()

    def forward(self, outputs, labels, *args, **kwargs):
        return 0
    
def format_loss(loss):
    return f"{loss:.3f}" if (loss > 0.01 or loss == 0.0) else f"{loss:.3e}"

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

        if self.loss_handler.run_teacher() and self.loss_handler.teacher_model is None:
            logger("Warning: teacher is None, but is specified to be run", col='yellow')

    def __call__(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_student_loss = 0.0
        running_vanilla_loss = 0.0
        running_kd_loss = 0.0
        num_batches = 0  # Track the number of processed batches
        all_labels = []
        all_preds = []
        top5_correct = 0
        total_samples = 0  # Keep track of the total number of samples

        for i, (inputs, labels, indices) in enumerate(self.trainloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            total_samples += labels.size(0)
            
            self.optimizer.zero_grad()

            teacher_logits = None
            if self.loss_handler.run_teacher():
                teacher_logits = self.loss_handler.teacher_model.generate_logits(inputs)

            with self.autocast(self.device.type):
                eval_losses = []
                outputs = self.model(inputs)
                student_loss = self.loss_handler.gamma * self.loss_handler.student_criterion(outputs, labels)

                vanilla_loss = torch.tensor(0.0, device=self.device)
                kd_loss = torch.tensor(0.0, device=self.device)
                if self.loss_handler.alpha != 0 and self.loss_handler.vanilla_criterion is not None:
                    vanilla_loss = self.loss_handler.alpha * self.loss_handler.vanilla_criterion(outputs, labels, teacher_logits, features=inputs, indices=indices)
                if self.loss_handler.beta != 0 and self.loss_handler.kd_criterion is not None:
                    kd_loss = self.loss_handler.beta * self.loss_handler.kd_criterion(outputs, labels, teacher_logits, features=inputs, indices=indices)

                loss = student_loss + vanilla_loss + kd_loss      
            
                # with torch.no_grad():
                #     self.model.eval()
                #     # process the eval criterions
                #     for crit_name, crit in self.loss_handler.eval_criterions:
                #         eval_losses.append((crit_name, crit(outputs, labels, teacher_logits, features=inputs, indices=indices)))
                #     self.model.train()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.loss_handler.batch_step(batch_idx=i)

            running_loss += loss.item()
            running_student_loss += student_loss.item()
            running_vanilla_loss += vanilla_loss.item()
            running_kd_loss += kd_loss.item()
            num_batches += 1  # Increment the number of processed batches
            
            _, predicted = torch.max(outputs, 1)
            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())

            # Calculate top-5 error
            top5 = outputs.topk(5, dim=1)[1]
            top5_correct += (top5 == labels.view(-1, 1)).sum().item()
            
            if i % 100 == 0:  # Log every 100 mini-batches
                mean_loss = running_loss / num_batches  # Calculate average loss over the batches so far
                mean_student_loss = running_student_loss / num_batches
                mean_vanilla_loss = running_vanilla_loss / num_batches
                mean_kd_loss = running_kd_loss / num_batches

                student_loss_str = f"Stu. Loss: {format_loss(mean_student_loss)}".ljust(23)
                vanilla_loss_str = "" if self.loss_handler.vanilla_criterion is None else f"Van. Loss: {format_loss(mean_vanilla_loss)}".ljust(23)
                kd_loss_str = "" if self.loss_handler.kd_criterion is None else f"KD Loss: {format_loss(mean_kd_loss)}".ljust(21)
                loss_str = f"Loss: {mean_loss:.3f}".ljust(15)
                batch_str = f"[Epoch {epoch}, Batch {i}/{len(self.trainloader) - 1}]".ljust(28)
                
                eval_losses = [f'{name}={eval_loss:.4e}' for (name, eval_loss) in eval_losses]
                eval_losses_str = "[" + " | ".join(eval_losses) + "]"

                self.logger(f'{batch_str} {loss_str} {student_loss_str} {vanilla_loss_str} {kd_loss_str} {eval_losses_str}')
                self.writer.add_scalar('training_loss', mean_loss, epoch * len(self.trainloader) + i)
                
        # Step the schedulers
        for scheduler in self.schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(running_loss / num_batches)  # Use the average running loss for ReduceLROnPlateau
            else:
                scheduler.step()
        
        self.loss_handler.epoch_step(epoch_idx=epoch)

        # Concatenate all predictions and labels for metrics calculation
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        # Calculate metrics
        epoch_loss = running_loss / num_batches
        accuracy = 100 * (all_preds == all_labels).sum().item() / len(all_labels)
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')
        top5_error = 100 * (1 - top5_correct / total_samples)

        self.writer.add_scalar('training_epoch_loss', epoch_loss, epoch)
        self.writer.add_scalar('training_accuracy', accuracy, epoch)
        self.writer.add_scalar('training_f1_score', f1, epoch)
        self.writer.add_scalar('training_top5_error', top5_error, epoch)

        log_entry = create_log_entry(epoch, 'train', epoch_loss, accuracy, f1, self.start_time, self.device, top5_error)
        self.logger.log_to_csv(log_entry)
