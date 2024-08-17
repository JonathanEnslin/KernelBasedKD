import torch.nn as nn

class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def run_teacher():
        raise NotImplementedError()
