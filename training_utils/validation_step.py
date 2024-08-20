import torch
import torch.cuda.amp as amp
from sklearn.metrics import f1_score
from utils.log_utils import create_log_entry, log_to_csv
from training_utils.evaluation_step import EvaluationStep

class ValidationStep(EvaluationStep):
    def __init__(self, model, valloader, criterion, device, writer, start_time, autocast, early_stopping=None, best_model_tracker=None, logger=print, no_write=False):
        super().__init__(model, valloader, criterion, device, writer, start_time, autocast, mode='vali', early_stopping=early_stopping, best_model_tracker=best_model_tracker, logger=logger, no_write=no_write)

    def __call__(self, epoch):
        return super().__call__(epoch)

