import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaKDLoss(nn.Module):
    def __init__(self, temperature=20, cached_teacher_logits=None):
        """
        Initializes the VanillaKDLoss module.
        
        Args:
        - alpha (float): Weight for the distillation loss.
        - temperature (float): Temperature for the distillation process.
        - teacher (nn.Module): Pre-trained teacher model (optional).
        """
        super(VanillaKDLoss, self).__init__()
        self.temperature = temperature
        self.cached_teacher_logits = cached_teacher_logits

    def forward(self, student_logits, labels, teacher_logits=None, features=None, indices=None):
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
        if teacher_logits is None and (self.cached_teacher_logits is None or indices is None):
            raise ValueError("Teacher logits or (indices and/or cached logits are not provided")

        teacher_logits = teacher_logits or self.cached_teacher_logits[indices]

        soft_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Combine losses
        return distillation_loss
        # loss = (self.alpha * distillation_loss) + ((1 - self.alpha) * student_loss)
        # return loss

