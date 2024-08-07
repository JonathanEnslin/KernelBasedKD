import torch
import torch.nn as nn


class KernelAttentionTransfer(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, alpha=1.0, beta=1.0):
        super(KernelAttentionTransfer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.alpha = alpha
        self.beta = beta

    def forward(self, student, teacher):
        student_attention = self.get_attention(student)
        teacher_attention = self.get_attention(teacher)
        loss = self.alpha * self.get_loss(student_attention, teacher_attention)
        return loss

    def get_attention(self, x):
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