'''
Sourced from https://github.com/Matthew-Dickson/FilterBasedKnowledgeDistillation/tree/main
'''

import torch
from torch import nn
from utils.amp_grad_scaling_handler import MockAutocast

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.feature_maps = []
        self.pre_activation_fmaps = []
        self.post_activation_fmaps = []
        self.all_conv_layers = []
        self.group_final_conv_layers = []
        self.hook_device_state = "cpu"
        self.autocast = MockAutocast
        self.device_type = "cpu"

    def to(self, device, *args, **kwargs):
        self.device_type = device.type
        return super(BaseModel, self).to(device, *args, **kwargs)

    def set_autocast(self, autocast):
        self.autocast = autocast

    def predict(self, dataloader, device="cpu"):
        self.eval()
        with torch.no_grad():
            correct = 0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                logits = self(images)
                _,predictions = torch.max(logits,1)
                correct += (predictions == labels).sum().item()

        return predictions, correct
    

    def set_hook_device_state(self, state):
        if state not in ["cpu", "cuda", "same"]:
            raise ValueError("Invalid device state. Must be either 'cpu', 'cuda' or 'same'")
        self.hook_device_state = state


    def get_hook_device_state(self):
        return self.hook_device_state


    def generate_logits(self, images):
        self.eval()
        with torch.no_grad():
            with self.autocast(self.device_type):
                logits = self(images)
        return logits
    

    def get_all_conv_layers(self):
        model = self
        conv_layers = []
        def recursive_find_conv_layers(module):
            for name, layer in module.named_children():
                if isinstance(layer, nn.Conv2d):
                    conv_layers.append(layer)
                elif len(list(layer.children())) > 0:
                    recursive_find_conv_layers(layer)
        recursive_find_conv_layers(model)
        return conv_layers


    def get_feature_maps(self, detached=False):
        if detached:
            return self._detach_list(self.feature_maps)
        return self.feature_maps


    def get_post_activation_fmaps(self, detached=False):
        if detached:
            return self._detach_list(self.post_activation_fmaps)
        return self.pre_activation_fmaps


    def _detach_list(self, list):
        if self.hook_device_state == "same":
            return [item.detach() for item in list]
        elif self.hook_device_state == "cpu":
            return [item.detach().cpu() for item in list]
            # return [item.cpu().detach() for item in list]
        else:
            return [item.detach().cuda() for item in list]
            # return [item.cuda().detach() for item in list]


    def get_pre_activation_fmaps(self, detached=False):
        if detached:
            return self._detach_list(self.pre_activation_fmaps)
        return self.post_activation_fmaps
    

    def _clear_feature_maps_lists(self):
        self.feature_maps = []
        self.pre_activation_fmaps = []
        self.post_activation_fmaps = []


    def save(self, path):
        torch.save(self.state_dict(), path)


    def get_group_final_kernel_weights(self, detached=True):
        weights = [kernel.weight for kernel in self.group_final_conv_layers]
        if detached:
            return self._detach_list(weights)
        return weights
        

    def get_all_kernel_weights(self, detached=True):
        weights = [kernel.weight for kernel in self.all_conv_layers]
        if detached:
            return self._detach_list(weights)
        return weights


    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
    
    