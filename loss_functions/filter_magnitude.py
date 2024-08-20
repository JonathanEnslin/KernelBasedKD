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
        self.detach_student = kwargs.get('detach_student', False)
        self.collect_indices = kwargs.get('collect_indices', 'all')
        self.pre_aggregate_decorators = kwargs.get('pre_aggregate_decorators', [])
        self.initial_aggregate_fn = kwargs.get('initial_aggregate_fn', partial(torch.mean, dim=(0, 1)))
        self.post_aggregate_decorators = kwargs.get('post_aggregate_decorators', [])
        self.post_flatten_decorators = kwargs.get('post_flatten_decorators', [])
        self.final_decorators = kwargs.get('final_decorators', [])
        self.final_aggregate_fn = kwargs.get('final_aggregate_fn', torch.mean)


    def forward(self, student_logits, teacher_logits, labels, features=None, indices=None):
        student_weights = self._get_model_weights(self.student, detached=self.detach_student)
        teacher_weights = self._get_model_weights(self.teacher, detached=True)

        if self.collect_indices != "all":
            student_weights = [student_weights[i] for i in self.collect_indices]
            teacher_weights = [teacher_weights[i] for i in self.collect_indices]

        collected_val = 0.0
        for s_w, t_w in zip(student_weights, teacher_weights):
            decorated_s_w = s_w
            decorated_t_w = t_w
            s_sign = torch.sign(s_w)
            t_sign = torch.sign(t_w)
            for decorator in self.pre_aggregate_decorators:
                if decorator == 'restore_signs':
                    decorated_s_w = s_sign * decorated_s_w
                    decorated_t_w = t_sign * decorated_t_w
                else:
                    decorated_s_w = decorator(decorated_s_w)
                    decorated_t_w = decorator(decorated_t_w)
            
            agg_s_w = self.initial_aggregate_fn(decorated_s_w)
            agg_t_w = self.initial_aggregate_fn(decorated_t_w)
            
            s_sign = torch.sign(agg_s_w)
            t_sign = torch.sign(agg_t_w)
            for decorator in self.post_aggregate_decorators:
                if decorator == 'restore_signs':
                    agg_s_w = s_sign * agg_s_w
                    agg_t_w = t_sign * agg_t_w
                else:
                    agg_s_w = decorator(agg_s_w)
                    agg_t_w = decorator(agg_t_w)

            # flatten
            agg_s_w = agg_s_w.view(-1)
            agg_t_w = agg_t_w.view(-1)

            s_sign = torch.sign(agg_s_w)
            t_sign = torch.sign(agg_t_w)
            for decorator in self.post_flatten_decorators:
                if decorator == 'restore_signs':
                    agg_s_w = s_sign * agg_s_w
                    agg_t_w = t_sign * agg_t_w
                else:
                    agg_s_w = decorator(agg_s_w)
                    agg_t_w = decorator(agg_t_w)

            diff = agg_s_w - agg_t_w

            for decorator in self.final_decorators:
                if decorator == 'restore_signs':
                    diff = s_sign * diff
                else:
                    diff = decorator(diff)

            collected_val += self.final_aggregate_fn(diff)

        return collected_val

    
    def _get_model_weights(self, model: BaseModel, detached=True):
        return model.get_group_final_kernel_weights(detached=detached)

    def run_teacher(self):
        return False