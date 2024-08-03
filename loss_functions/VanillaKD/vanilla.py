import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaKDLoss(nn.Module):
    def __init__(self, alpha=0.1, temperature=20, teacher=None):
        """
        Initializes the VanillaKDLoss module.
        
        Args:
        - alpha (float): Weight for the distillation loss.
        - temperature (float): Temperature for the distillation process.
        - teacher (nn.Module): Pre-trained teacher model (optional).
        """
        super(VanillaKDLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.teacher = teacher

    def forward(self, student_logits, labels, teacher_logits=None, features=None):
        """
        Forward pass for computing the KD loss.

        Args:
        - student_logits (torch.Tensor): Logits from the student model.
        - labels (torch.Tensor): Ground truth labels.
        - teacher_logits (torch.Tensor): Logits from the teacher model (optional).
        - features (torch.Tensor): Input features/images (optional, if teacher model is provided).

        Returns:
        - loss (torch.Tensor): Combined KD and cross-entropy loss.
        """
        if teacher_logits is None:
            if self.teacher is None:
                raise ValueError("Teacher model is not provided")
            if features is None:
                raise ValueError("Features are not provided")
            teacher_logits = self.teacher(features)            

        soft_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        student_loss = F.cross_entropy(student_logits, labels)
        
        # Combine losses
        loss = (self.alpha * distillation_loss) + ((1 - self.alpha) * student_loss)
        return loss

