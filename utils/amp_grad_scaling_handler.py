import torch

class MockGradScaler:
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass

class MockAutocast:
    def __init__(self, device_type):
        pass
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass


def get_amp_and_scaler(args, device, logger=print):
    if not args.use_amp:
        return MockGradScaler(), MockAutocast
    else:
        try:
            return torch.GradScaler(device.type), torch.autocast
        except:
            logger("Warning: Could not return GradScaler and autocast context manager. Returning mock classes instead.")
            return MockGradScaler(), MockAutocast
    

