import torch
import os

def save_checkpoint(state, is_best, filename):
    print(f"=> Saving checkpoint '{filename}'. Please do not interrupt the saving process else the checkpoint file might get corrupted.")
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'model_best.pth.tar')
    print(f"=> Checkpoint saved at '{filename}'")

def load_checkpoint(model, optimizer, schedulers, scaler, filename):
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for scheduler, state_dict in zip(schedulers, checkpoint['schedulers']):
            scheduler.load_state_dict(state_dict)
        scaler.load_state_dict(checkpoint['scaler'])
        print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return start_epoch
    else:
        print(f"=> no checkpoint found at '{filename}'")
        return 0
