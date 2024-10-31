import torch

def initialize_model(model_name, num_classes, device, logger=print, **kwargs):
    if model_name == 'resnet8':
        from models.resnet import resnet8
        model = resnet8(num_classes=num_classes, **kwargs).to(device)
    elif model_name == 'resnet20':
        from models.resnet import resnet20
        model = resnet20(num_classes=num_classes, **kwargs).to(device)
    elif model_name == 'resnet32':
        from models.resnet import resnet32
        model = resnet32(num_classes=num_classes, **kwargs).to(device)
    elif model_name == 'resnet56':
        from models.resnet import resnet56
        model = resnet56(num_classes=num_classes, **kwargs).to(device)
    elif model_name == 'resnet110':
        from models.resnet import resnet110
        model = resnet110(num_classes=num_classes, **kwargs).to(device)
    elif model_name == 'dummy_student':
        from models.dummy_student_model import DummyStudentModel
        model = DummyStudentModel(3, 100).to(device)
    elif model_name == 'dummy_teacher':
        from models.dummy_teacher_model import DummyTeacherModel
        model = DummyTeacherModel(3, 100).to(device)
    elif model_name == 'resnet20x2':
        from models.resnet import resnet20x2
        model = resnet20x2(num_classes=num_classes, **kwargs).to(device)
    elif model_name == 'resnet8x4':
        from models.resnet import resnet8x4
        model = resnet8x4(num_classes=num_classes, **kwargs).to(device)
    elif model_name == 'resnet20x4':
        from models.resnet import resnet20x4
        model = resnet20x4(num_classes=num_classes, **kwargs).to(device)
    elif model_name == 'resnet32x4':
        from models.resnet import resnet32x4
        model = resnet32x4(num_classes=num_classes, **kwargs).to(device)
    else:
        logger(f"Unknown model name {model_name}")
        return None
    return model


def get_optimizer(params, model):
    optimizer_type = params['optimizer']['type']
    optimizer_kwargs = params['optimizer']['parameters']
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer type {optimizer_type}")
    return optimizer


def get_schedulers(params, optimizer):
    schedulers = []
    for scheduler_params in params.get('schedulers', []):
        scheduler_type = scheduler_params['type']
        scheduler_kwargs = scheduler_params['parameters']
        if scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
        elif scheduler_type == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_kwargs)
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler type {scheduler_type}")
        schedulers.append(scheduler)
    return schedulers
