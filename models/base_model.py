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
        self.hook_device_state = "cpu"
        self.autocast = MockAutocast

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
            with self.autocast(self.device.type):
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

    
    def _feature_map_hook_fn(self, module, input, output):
        with torch.no_grad():
            cnn_out = output.detach()
            if self.hook_device_state == "same":
                self.feature_maps.append(cnn_out)
            elif self.hook_device_state == "cpu":
                self.feature_maps.append(cnn_out.cpu())
            elif self.hook_device_state == "cuda":
                self.feature_maps.append(cnn_out.cuda())
            del cnn_out
    

    def _group_preact_fmap_hook_fn(self, module, input, output):
        with torch.no_grad():
            cnn_out = output.detach()
            if self.hook_device_state == "same":
                self.post_activation_fmaps.append(cnn_out)
            elif self.hook_device_state == "cpu":
                self.post_activation_fmaps.append(cnn_out.cpu())
            elif self.hook_device_state == "cuda":
                self.post_activation_fmaps.append(cnn_out.cuda())
            del cnn_out

    def _group_postact_fmap_hook_fn(self, module, input, output):
        with torch.no_grad():
            cnn_out = output.detach()
            if self.hook_device_state == "same":
                self.pre_activation_fmaps.append(cnn_out)
            elif self.hook_device_state == "cpu":
                self.pre_activation_fmaps.append(cnn_out.cpu())
            elif self.hook_device_state == "cuda":
                self.pre_activation_fmaps.append(cnn_out.cuda())
            del cnn_out


    def get_feature_maps(self):
        return self.feature_maps


    def get_post_activation_fmaps(self):
        return self.pre_activation_fmaps


    def get_pre_activation_fmaps(self):
        return self.post_activation_fmaps
    

    def _clear_feature_maps_lists(self):
        self.feature_maps = []
        self.pre_activation_fmaps = []
        self.post_activation_fmaps = []


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
    
    