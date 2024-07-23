'''
Adapted from https://github.com/Matthew-Dickson/FilterBasedKnowledgeDistillation/tree/main
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class VanillaKDLoss(nn.Module):
    def __init__(self, alpha=0.1, temperature=20):
        """
        Initializes the VanillaKDLoss module.
        
        Args:
        - alpha (float): Weight for the distillation loss.
        - temperature (float): Temperature for the distillation process.
        """
        super(VanillaKDLoss, self).__init__()
        self.alpha =  alpha
        self.temperature = temperature

    def forward(self, student_logits, features, labels, teacher_model: BaseModel):
        """
        Forward pass for computing the KD loss.

        Args:
        - student_logits (torch.Tensor): Logits from the student model.
        - features (torch.Tensor): Input features/images.
        - labels (torch.Tensor): Ground truth labels.
        - teacher_model (BaseModel): Pre-trained teacher model.

        Returns:
        - loss (torch.Tensor): Combined KD and cross-entropy loss.
        """
        teacher_logits = teacher_model.generate_logits(images=features)
        # log_softmax is used to calculate the log probabilities of the student's outputs
        #  since kl_div expects log probabilities for inputs
        soft_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean')
        student_loss = F.cross_entropy(student_logits, labels)
        # Combine
        loss = (self.alpha * distillation_loss)  + ((1-self.alpha) * student_loss) 
        return loss

