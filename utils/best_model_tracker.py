import numpy as np
import torch

class BestModelTracker:
    """Tracks the best validation model after a given number of epochs."""
    def __init__(self, verbose=False, delta=0, path='best_model.pt', trace_func=print, monitor='loss', enabled_after_epoch=0):
        """
        Args:
            verbose (bool): If True, prints a message for each validation metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'best_model.pt'
            trace_func (function): trace print function.
                            Default: print
            monitor (str): Whether to monitor 'loss' or 'accuracy'.
                            Default: 'loss'
            enabled_after_epoch (int): Number of epochs to wait before starting tracking the best model.
                            Default: 0
        """
        self.verbose = verbose
        self.best_score = None
        self.val_metric_min = np.inf if monitor == 'loss' else -np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.monitor = monitor
        self.enabled_after_epoch = enabled_after_epoch
        self.epoch = 0

    def __call__(self, val_metric, model):
        self.epoch += 1
        
        # Only start tracking the best model after enabled_after_epoch
        if self.epoch < self.enabled_after_epoch:
            return

        score = -val_metric if self.monitor == 'loss' else val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_metric, model)

    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation metric improves.'''
        if self.verbose:
            self.trace_func(f'Validation {self.monitor} {"decreased" if self.monitor == "loss" else "increased"} '
                            f'({self.val_metric_min:.6f} --> {val_metric:.6f}).  Saving best model ...')
        torch.save(model.state_dict(), self.path)
        self.val_metric_min = val_metric
