import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel


class FilterAttentionTransfer(BaseModel):
    def __init__(self, student: BaseModel, teacher: BaseModel, map_p=2, loss_p=2, mean_targets=['C_out', 'C_in'], use_abs=False, layer_groups='final'):
        super(FilterAttentionTransfer, self).__init__()
        # self.get_attention_maps = self._semi_flatten_abs_to_p
        # self.get_attention_maps = self._no_flatten_abs_to_p
        # self.get_attention_maps = self._flatten_abs_to_p
        # self.get_attention_maps = self._flatten_cout_abs_to_p

        if layer_groups not in ['final', 'all']:
            raise ValueError("Invalid layer_groups. Must be either 'final' or 'all'")

        self.layer_groups = layer_groups

        if mean_targets == ['C_out', 'C_in']:
            self.get_attention_maps = self._flatten_abs_to_p if use_abs else self._flatten_no_abs_no_p
        elif mean_targets == ['C_out']:
            self.get_attention_maps = self._flatten_cout_abs_to_p if use_abs else self._flatten_cout_no_abs_no_p
        elif mean_targets == ['C_in']:
            self.get_attention_maps = self._flatten_cin_abs_to_p if use_abs else self._flatten_cin_no_abs_no_p
        elif len(mean_targets) == 0:
            self.get_attention_maps = self._no_flatten_abs_to_p if use_abs else self._no_flatten_no_abs_no_p
        else:
            raise ValueError("Invalid mean_targets. Must be either ['C_out', 'C_in'], ['C_out'], ['C_in'], or []")
        
        self.calc_loss = self._compute_f_at_loss_pow
        self.student = student
        self.teacher = teacher
        self.map_p = map_p
        self.loss_p = loss_p
        # self.get_model_weights = self._get_selective_model_weights
        if self.layer_groups == 'final':
            self.get_model_weights = self._get_group_final_model_weights
        elif self.layer_groups == 'all':
            self.get_model_weights = self._get_all_weights


    def forward(self, student_logits, teacher_logits, labels, features=None, indices=None):
        student_weights = self.get_model_weights(self.student, detached=False)
        teacher_weights = self.get_model_weights(self.teacher, detached=True)
        # student_weights = self._get_group_final_model_weights(self.student, detached=False)
        # teacher_weights = self._get_group_final_model_weights(self.teacher, detached=True)

        loss = 0
        for student_filter, teacher_filter in zip(student_weights, teacher_weights):
            student_attention = self.get_attention_maps(student_filter, self.map_p)
            teacher_attention = self.get_attention_maps(teacher_filter, self.map_p)
            loss += self.calc_loss(student_attention, teacher_attention, self.loss_p)

        return loss


    def run_teacher(self):
        return False


    def _get_group_final_model_weights(self, model: BaseModel, detached=True):
        return model.get_group_final_kernel_weights(detached=detached)
        

    def _get_all_weights(self, model: BaseModel, detached=True):
        return model.get_all_kernel_weights(detached=detached)


    def _get_selective_model_weights(self, model: BaseModel, detached=True):
        # EXPERIMENTAL FUNCTION
        all_weights = model.get_all_kernel_weights(detached=detached)
        if len(all_weights) == 55:
            # resnet56
            indices = [0, 1, 7, 13, 19, 25, 31, 37, 43, 49]
            return [all_weights[i] for i in indices[::2]]
        if len(all_weights) == 19:
            # resnet20
            indices = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17]
            return [all_weights[i] for i in indices[::2]]
            

    def _flatten_cin_abs_to_p(self, filter_weights, p):
        # Filter weights have dimensions (C_out, C_in, H, W)
        # This method returns a tensor of shape (C_out, H * W)
        # And computes attention across C_in for each filter

        # First, we take the absolute value of the filter weights
        filter_weights = torch.abs(filter_weights)
        # Next we raise the filter weights to the power of p
        filter_weights = torch.pow(filter_weights, p)
        # Next we sum across the C_in dimension
        filter_weights = torch.mean(filter_weights, dim=1)
        # Next we flatten the filter weights
        filter_weights = filter_weights.view( -1)
        normalized_filter_weights = F.normalize(filter_weights, dim=0, eps=1e-6)
        return normalized_filter_weights


    def _no_flatten_abs_to_p(self, filter_weights, p):
        # Filter weights have dimensions (C_out, C_in, H, W)

        # First, we take the absolute value of the filter weights
        filter_weights = torch.abs(filter_weights)
        # Next we raise the filter weights to the power of p
        filter_weights = torch.pow(filter_weights, p)
        # Next we flatten the filter weights
        filter_weights = filter_weights.view( -1)
        normalized_filter_weights = F.normalize(filter_weights, dim=0, eps=1e-6)
        return normalized_filter_weights
    
    
    def _flatten_cout_abs_to_p(self, filter_weights, p):
        # Filter weights have dimensions (C_out, C_in, H, W)
        # This method returns a tensor of shape (C_in, H * W)
        # And computes attention across C_out for each filter

        # First, we take the absolute value of the filter weights
        filter_weights = torch.abs(filter_weights)
        # Next we raise the filter weights to the power of p
        filter_weights = torch.pow(filter_weights, p)
        # Next we sum across the C_out dimension
        filter_weights = torch.mean(filter_weights, dim=0)
        # Next we flatten the filter weights
        filter_weights = filter_weights.view( -1)
        normalized_filter_weights = F.normalize(filter_weights, dim=0, eps=1e-6)
        return normalized_filter_weights


    def _flatten_abs_to_p(self, filter_weights, p, eps=1e-6):
        # Filter weights have dimensions (C_out, C_in, H, W)
        # This method returns a tensor of shape (H * W)
        # And computes attention across C_in and C_out for each filter
        f_at = filter_weights.abs()
        f_at = f_at.pow(p)
        f_at = f_at.mean(dim=(0, 1))
        f_at = f_at.view(-1)
        normalized_f_at = F.normalize(f_at, dim=0, eps=eps)
        return normalized_f_at


    def _flatten_cin_no_abs_no_p(self, filter_weights, p, eps=1e-6):
        # Filter weights have dimensions (C_out, C_in, H, W)
        # This method returns a tensor of shape (C_out, H * W)
        # And computes attention across C_in for each filter

        # We sum across the C_in dimension
        filter_weights = torch.mean(filter_weights, dim=1)
        # Next we flatten the filter weights
        filter_weights = filter_weights.view(-1)
        normalized_filter_weights = F.normalize(filter_weights, dim=0, eps=1e-6)
        return normalized_filter_weights
        

    def _no_flatten_no_abs_no_p(self, filter_weights, p, eps=1e-6):
        # Filter weights have dimensions (C_out, C_in, H, W)

        # We flatten the filter weights
        filter_weights = filter_weights.view(-1)
        normalized_filter_weights = F.normalize(filter_weights, dim=0, eps=1e-6)
        return normalized_filter_weights


    def _flatten_cout_no_abs_no_p(self, filter_weights, p, eps=1e-6):
        # Filter weights have dimensions (C_out, C_in, H, W)
        # This method returns a tensor of shape (C_in, H * W)
        # And computes attention across C_out for each filter

        # We sum across the C_out dimension
        filter_weights = torch.mean(filter_weights, dim=0)
        # Next we flatten the filter weights
        filter_weights = filter_weights.view(-1)
        normalized_filter_weights = F.normalize(filter_weights, dim=0, eps=1e-6)
        return normalized_filter_weights


    def _flatten_no_abs_no_p(self, filter_weights, p, eps=1e-6):
        # Filter weights have dimensions (C_out, C_in, H, W)
        # This method returns a tensor of shape (H * W)
        # And computes attention across C_in and C_out for each filter
        
        f_at = filter_weights.mean(dim=(0, 1))
        f_at = f_at.view(-1)
        normalized_f_at = F.normalize(f_at, dim=0, eps=eps)
        return normalized_f_at


    def _compute_f_at_loss_pow(self, student_f_at, teacher_f_at, p=1):
        return (student_f_at - teacher_f_at).pow(p).mean()


    def get_loss(self, student_attention, teacher_attention):
        loss = torch.mean(torch.abs(student_attention - teacher_attention))
        return loss
    
def get_group_boundary_indices_for_resnet(resnet_model: str):
    raise NotImplementedError()
