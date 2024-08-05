'''
Sourced from https://github.com/Matthew-Dickson/FilterBasedKnowledgeDistillation/tree/main
'''

import torch
from torch import nn
from functions.activation.activation_functions import batch_softmax_with_temperature

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.feature_maps = []
        self.layer_group_output_feature_maps = []
        self.layer_group_preactivation_feature_maps = []
        self.hook_device_state = "cpu"
        

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

    def generate_soft_targets(self, images, temperature = 40):
        self.eval()
        with torch.no_grad():
            logits = self(images)
            probs_with_temperature = batch_softmax_with_temperature(logits, temperature)
        return probs_with_temperature


    def generate_logits(self, images):
        self.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
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
    

    def _group_preactivation_hook_fn(self, module, input, output):
        with torch.no_grad():
            cnn_out = output.detach()
            if self.hook_device_state == "same":
                self.layer_group_preactivation_feature_maps.append(cnn_out)
            elif self.hook_device_state == "cpu":
                self.layer_group_preactivation_feature_maps.append(cnn_out.cpu())
            elif self.hook_device_state == "cuda":
                self.layer_group_preactivation_feature_maps.append(cnn_out.cuda())
            del cnn_out

    def _group_output_hook_fn(self, module, input, output):
        with torch.no_grad():
            cnn_out = output.detach()
            if self.hook_device_state == "same":
                self.layer_group_output_feature_maps.append(cnn_out)
            elif self.hook_device_state == "cpu":
                self.layer_group_output_feature_maps.append(cnn_out.cpu())
            elif self.hook_device_state == "cuda":
                self.layer_group_output_feature_maps.append(cnn_out.cuda())
            del cnn_out


    def get_feature_maps(self):
        return self.feature_maps


    def get_layer_group_output_feature_maps(self):
        return self.layer_group_output_feature_maps


    def get_layer_group_preactivation_feature_maps(self):
        return self.layer_group_preactivation_feature_maps
    

    def _clear_feature_maps_lists(self):
        self.feature_maps = []
        self.layer_group_output_feature_maps = []
        self.layer_group_preactivation_feature_maps = []


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
    
    