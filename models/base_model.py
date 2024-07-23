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
    

    def generate_soft_targets(self, images, temperature = 40):
        self.eval()
        with torch.no_grad():
            logits = self(images)
            probs_with_temperature = batch_softmax_with_temperature(logits, temperature)
        return probs_with_temperature
    

    def generate_logits(self, images):
        self.eval()
        with torch.no_grad():
            logits = self(images)
        return logits

    def _feature_map_hook_fn(self, module, input, output):
        with torch.no_grad():
            cnn_out = output.detach()
            self.feature_maps.append(cnn_out.cpu())
            del cnn_out
    

    def get_feature_maps(self):
        return self.feature_maps
    

    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
    
    