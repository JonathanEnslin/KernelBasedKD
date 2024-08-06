import torch

class MockGradScaler:
    def __init__(self, *args, **kwargs):
        pass

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    def update(self, *args, **kwargs):
        pass

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}

class MockAutocast:
    def __init__(self, device_type):
        pass
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass


def get_amp_and_grad_scaler(args, device, logger=print):
    '''
    Returns the appropriate GradScaler and autocast context manager based on whether AMP is enabled or not.
    If AMP is not enabled, returns mock classes that do nothing
    returns: GradScaler, autocast
    '''
    if not args.use_amp:
        return MockGradScaler(), MockAutocast
    else:
        try:
            return torch.GradScaler(device.type), torch.autocast
        except:
            logger("Warning: Could not return GradScaler and autocast context manager. Returning mock classes instead.")
            return MockGradScaler(), MockAutocast
    

