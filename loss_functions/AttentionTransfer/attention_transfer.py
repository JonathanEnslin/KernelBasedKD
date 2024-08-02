'''
Adapted from:
https://github.com/yoshitomo-matsubara/torchdistill/blob/db102321a0c7fbe94b88465332dd99e8f10d18e5/torchdistill/losses/mid_level.py#L315
and
https://github.com/szagoruyko/attention-transfer
and
https://github.com/AlexandrosFerles/NIPS_2019_Reproducibilty_Challenge_Zero-shot_Knowledge_Transfer_via_Adversarial_Belief_Matching/blob/master/src/PyTorch/utils.py
and
https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/at.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModelatten


## ============= FROM TORCH DISTILL, TWO METHODS, PAPER VERSION AND IMPL VERSION =============
# ---- for constructing the attention maps using the paper method ----
# @staticmethod
# def attention_transfer_paper(feature_map):
#     return normalize(feature_map.pow(2).sum(1).flatten(1))

# ---- for computing the loss using the paper method ----
# def compute_at_loss_paper(self, student_feature_map, teacher_feature_map):
#     at_student = self.attention_transfer_paper(student_feature_map)
#     at_teacher = self.attention_transfer_paper(teacher_feature_map)
#     return torch.norm(at_student - at_teacher, dim=1).sum()

# ---- for cosntructing the attention maps using the implementation method ----
# @staticmethod
# def attention_transfer(feature_map):
#     return normalize(feature_map.pow(2).mean(1).flatten(1))

# ---- for computing the loss using the implementation method ----
# def compute_at_loss(self, student_feature_map, teacher_feature_map):
#     at_student = self.attention_transfer(student_feature_map)
#     at_teacher = self.attention_transfer(teacher_feature_map)
#     return (at_student - at_teacher).pow(2).mean()


