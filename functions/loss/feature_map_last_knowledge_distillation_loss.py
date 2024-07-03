'''
Sourced from https://github.com/Matthew-Dickson/FilterBasedKnowledgeDistillation/tree/main
'''

import torch
import torch.nn as nn
from torch.nn.functional import softmax

class FeatureMapLastKnowledgeDistillationLoss(nn.Module):
    def __init__(self,
                 options,
                 distillation_criterion = nn.KLDivLoss(reduction='batchmean'),
                 student_criterion = nn.CrossEntropyLoss()):
        super(FeatureMapLastKnowledgeDistillationLoss, self).__init__()
        self.alpha =  options.get("alpha") if options.get("alpha") != None else 0.1    
        self.beta =  options.get("beta") if options.get("beta") != None else 1     
        self.temperature =  options.get("temperature") if options.get("temperature") != None else 20     

        self.distillation_criterion = distillation_criterion
        self.student_criterion= student_criterion

    
    def _get_feature_maps(self, model):
        feature_maps =  model.get_feature_maps()
        flatten_feature_maps = []
        for map in feature_maps:
            flatten_feature_maps.append(torch.flatten(map))
        return flatten_feature_maps
    
    def _batch_softmax_with_temperature(self, batch_logits, temperature) -> torch.tensor:
        return softmax(batch_logits/temperature,dim=1)



    def _calculate_feature_map_loss(self, student_model, teacher_model, features):
        last_student_feature_map = self._get_feature_maps(student_model)[-1]
        last_teacher_feature_map = self._get_feature_maps(teacher_model)[-1]

        distance = torch.sqrt(torch.sum(torch.square(last_teacher_feature_map)) - torch.sum(torch.square(last_student_feature_map)))
        return distance


    def forward(self, student_logits, features, labels, student_model, teacher_model):
        soft_probabilities = self._batch_softmax_with_temperature(batch_logits=student_logits, temperature=self.temperature)
        soft_targets = teacher_model.generate_soft_targets(images = features, temperature = self.temperature)
        distillation_loss = self.distillation_criterion(soft_probabilities, soft_targets) 
        student_loss = self.student_criterion(student_logits, labels) 
        feature_map_loss = self._calculate_feature_map_loss(student_model, teacher_model,features) 
        loss = ((1-self.beta)*(self.alpha * distillation_loss  + (1-self.alpha) * student_loss)) + (self.beta* feature_map_loss)
        return loss