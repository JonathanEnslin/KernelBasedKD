import torch
import torch.nn as nn
from models.resnet import resnet20, resnet56, resnet110  # Adjust import according to your setup

def initialize_model(model_name, device):
    if model_name == 'resnet20':
        model = resnet20()
    elif model_name == 'resnet56':
        model = resnet56()
    elif model_name == 'resnet110':
        model = resnet110()
    else:
        raise ValueError(f"Unknown model name {model_name}")

    model.to(device)
    return model

def get_optimizer(params, model):
    optimizer_params = params['optimizer']['parameters']
    if params['optimizer']['type'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer type {params['optimizer']['type']}")
    return optimizer

def get_scheduler(params, optimizer):
    scheduler_params = params['scheduler']['parameters']
    if params['scheduler']['type'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif params['scheduler']['type'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
    elif params['scheduler']['type'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown scheduler type {params['scheduler']['type']}")
    return scheduler
