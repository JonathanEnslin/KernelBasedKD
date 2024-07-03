'''
Sourced from https://github.com/Matthew-Dickson/FilterBasedKnowledgeDistillation/tree/main
'''

import torch
import torch.nn as nn
from torch.nn.functional import softmax

class TraditionalKnowledgeDistillationLoss(nn.Module):
    def __init__(self,
                 options,
                 distillation_criterion = nn.KLDivLoss(reduction='batchmean'),
                 student_criterion = nn.CrossEntropyLoss()):
        super(TraditionalKnowledgeDistillationLoss, self).__init__()
        self.alpha =  options.get("alpha") if options.get("alpha") != None else 0.1   
        self.temperature =  options.get("temperature") if options.get("temperature") != None else 20   
        self.distillation_criterion = distillation_criterion
        self.student_criterion= student_criterion

    
    def _batch_softmax_with_temperature(self, batch_logits, temperature) -> torch.tensor:
        return softmax(batch_logits/temperature,dim=1)

    def forward(self, student_logits, features, labels, teacher_model):
        soft_probabilities = self._batch_softmax_with_temperature(batch_logits=student_logits, temperature=self.temperature)
        soft_targets = teacher_model.generate_soft_targets(images = features,
                                                                temperature = self.temperature)
        distillation_loss = self.distillation_criterion(soft_probabilities, soft_targets) 
        student_loss = self.student_criterion(student_logits, labels) 
        loss = (self.alpha * distillation_loss)  + ((1-self.alpha) * student_loss) 
        return loss
