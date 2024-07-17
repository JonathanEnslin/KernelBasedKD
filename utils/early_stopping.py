import numpy as np

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print, monitor='loss', enabled_after_epoch=0):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation metric improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
            monitor (str): Whether to monitor 'loss' or 'accuracy'.
                            Default: 'loss'
            enabled_after_epoch (int): Number of epochs to wait before starting early stopping.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf if monitor == 'loss' else -np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.monitor = monitor
        self.enabled_after_epoch = enabled_after_epoch
        self.epoch = 0

    def __call__(self, val_metric):
        self.epoch += 1
        
        # Only start checking early stopping criteria after enabled_after_epoch
        if self.epoch < self.enabled_after_epoch:
            return

        score = -val_metric if self.monitor == 'loss' else val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
