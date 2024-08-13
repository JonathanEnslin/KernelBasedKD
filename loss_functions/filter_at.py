import torch
import torch.nn as nn


class FilterAttentionTransfer(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, alpha=1.0, beta=1.0):
        super(FilterAttentionTransfer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.alpha = alpha
        self.beta = beta
        self.get_attention = self._get_attention_huh_what

    def forward(self, student, teacher):
        student_attention = self.get_attention(student)
        teacher_attention = self.get_attention(teacher)
        loss = self.alpha * self.get_loss(student_attention, teacher_attention)
        return loss

    def _semi_flatten_abs_to_p(self, filter_weights, p):
        # Filter weights have dimensions (C_out, C_in, H, W)
        # This method returns a tensor of shape (C_out, H * W)
        # And computes attention across C_in for each filter

        # First, we take the absolute value of the filter weights
        filter_weights = torch.abs(filter_weights)
        # Next we raise the filter weights to the power of p
        filter_weights = torch.pow(filter_weights, p)
        # Next we sum across the C_in dimension
        filter_weights = torch.sum(filter_weights, dim=1)
        # Next we flatten the filter weights
        filter_weights = filter_weights.view(filter_weights.size(0), -1)
        raise NotImplementedError()

    def _flatten_abs_to_p(self, filter_weights, p):
        # Filter weights have dimensions (C_out, C_in, H, W)
        # This method returns a tensor of shape (H * W)
        # And computes attention across C_in and C_out for each filter
        

        # First, we take the absolute value of the filter weights
        filter_weights = torch.abs(filter_weights)
        # Next we raise the filter weights to the power of p
        filter_weights = torch.pow(filter_weights, p)
        # Next we sum across the C_in dimension
        filter_weights = torch.sum(filter_weights, dim=1)
        # Next we flatten the filter weights
        filter_weights = filter_weights.view(filter_weights.size(0), -1)
        return filter_weights

    def _get_attention_huh_what(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        x = x.view(B * C, 1, H, W)
        attention = torch.nn.functional.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        attention = attention.view(B, C, -1)
        attention = torch.nn.functional.softmax(attention, dim=2)
        return attention

    def get_loss(self, student_attention, teacher_attention):
        loss = torch.mean(torch.abs(student_attention - teacher_attention))
        return loss