import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from functools import partial

# !!! NOT AN ACTUAL LOSS FUNCTION !!!
# Used for analysis purposes only

def nop(arg):
    return arg

class FilterAnalysis(BaseModel):
    def __init__(self, student: BaseModel, teacher: BaseModel, **kwargs):
        super(FilterAnalysis, self).__init__()
        self.student = student
        self.teacher = teacher
        self.collect_indices = kwargs.get('collect_indices', 'all')
        self.pre_aggregate_decorators = kwargs.get('pre_aggregate_decorators', [])
        self.target_model = self.student if kwargs.get('target_model', 'student') == 'student' else self.teacher
        self.initial_aggregate_fn = kwargs.get('initial_aggregate_fn', partial(torch.mean, dim=(0, 1)))
        self.post_aggregate_decorators = kwargs.get('post_aggregate_decorators', [])
        self.final_decorators = kwargs.get('final_decorators', [])
        self.final_aggregate_fn = kwargs.get('final_aggregate_fn', torch.mean)


    def forward(self, student_logits, teacher_logits, labels, features=None, indices=None):
        model_weights = self._get_model_weights(self.target_model, detached=True)

        if self.collect_indices != "all":
            model_weights = [model_weights[i] for i in self.collect_indices]

        collected_val = 0.0
        for weight in model_weights:
            decorated_weights = weight
            for decorator in self.pre_aggregate_decorators:
                decorated_weights = decorator(decorated_weights)
            
            aggregated_weights = self.initial_aggregate_fn(decorated_weights)
            
            for decorator in self.post_aggregate_decorators:
                aggregated_weights = decorator(aggregated_weights)

            # flatten
            aggregated_weights = aggregated_weights.view(-1)

            for decorator in self.final_decorators:
                aggregated_weights = decorator(aggregated_weights)

            collected_val += self.final_aggregate_fn(aggregated_weights)

        return collected_val

    
    def _get_model_weights(self, model: BaseModel, detached=True):
        return model.get_group_final_kernel_weights(detached=detached)

    def run_teacher(self):
        return False