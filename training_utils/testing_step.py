import torch
import torch.cuda.amp as amp
from sklearn.metrics import f1_score
from utils.log_utils import create_log_entry
from training_utils.evaluation_step import EvaluationStep

class TestStep(EvaluationStep):
    def __init__(self, model, testloader, criterion, device, writer, start_time, autocast, logger=print, no_write=False):
        super().__init__(model, testloader, criterion, device, writer, start_time, autocast, mode='test', logger=logger, no_write=no_write)

    def __call__(self, epoch):
        _, accuracy = super().__call__(epoch)
        return accuracy

