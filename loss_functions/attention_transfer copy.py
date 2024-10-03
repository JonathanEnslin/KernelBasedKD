import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
import models.resnet
from loss_functions.base_loss_fn import BaseLoss

'''
Adapted from:
https://github.com/yoshitomo-matsubara/torchdistill/blob/db102321a0c7fbe94b88465332dd99e8f10d18e5/torchdistill/losses/mid_level.py#L315
and
https://github.com/szagoruyko/attention-transfer
and
https://github.com/AlexandrosFerles/NIPS_2019_Reproducibilty_Challenge_Zero-shot_Knowledge_Transfer_via_Adversarial_Belief_Matching/blob/master/src/PyTorch/utils.py
and
https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/at.py
'''


## ============= FROM TORCH DISTILL, TWO METHODS, PAPER VERSION AND IMPL VERSION =============
# ---- for constructing the attention maps using the paper method ----
# @staticmethod
# def attention_transfer_paper(feature_map):
#     return normalize(feature_map.pow(2).sum(1).flatten(1))

# ---- for computing the loss using the paper method ----
# def compute_at_loss_paper(self, student_feature_map, teacher_feature_map):
#     at_student = self.attention_transfer_paper(student_feature_map)
#     at_teacher = self.attention_transfer_paper(teacher_feature_map)
#     return torch.norm(at_student - at_teacher, dim=1).sum()

# ---- for cosntructing the attention maps using the implementation method ----
# @staticmethod
# def attention_transfer(feature_map):
#     return normalize(feature_map.pow(2).mean(1).flatten(1))

# ---- for computing the loss using the implementation method ----
# def compute_at_loss(self, student_feature_map, teacher_feature_map):
#     at_student = self.attention_transfer(student_feature_map)
#     at_teacher = self.attention_transfer(teacher_feature_map)
#     return (at_student - at_teacher).pow(2).mean()

# Standard layer indices are as follows:
# As the paper specifies, feature maps at the end of each residual block are used
# For resnet20: [6, 12, 18]
# For resnet32: [10, 20, 30]
# For resnet56: [18, 36, 54]
# For resnet110: [36, 72, 108]


class ATLoss(BaseLoss):
    def __init__(self, student: BaseModel, teacher: BaseModel, mode='impl',
                  cached_pre_activation_fmaps=None, cached_post_activation_fmaps=None,
                    device='cuda', use_post_activation=None):
        """
        Initializes the ATLoss module.
        
        Args:
        - student (BaseModel): The student model.
        - teacher (BaseModel): The teacher model.
        - beta (float): Weight for the attention transfer loss.
        - mode (str): Mode for attention map and loss computation.
        - cached_pre_activation_fmaps (dict): Cached pre-activation feature maps.
        - cached_post_activation_fmaps (dict): Cached post-activation feature maps.
        - device (str): Device to use ('cuda' or 'cpu').
        - use_post_activation (bool): Whether to use post-activation feature maps.
        """
        super(ATLoss, self).__init__()
        self.student: BaseModel = student
        self.teacher: BaseModel = teacher
        self.standard_criterion = nn.CrossEntropyLoss()
        self.mode = mode
        self.device = device
        self.cached_teacher_maps = None
        self.beta = 1.0

        pre_activation_methods = ['impl', 'paper', 'paper_strict']
        post_activation_methods = ['zoo']

        # Organize the feature map getters
        if mode in pre_activation_methods:
            self.cached_teacher_maps = cached_pre_activation_fmaps
            self.non_cached_feature_map_getter = self._get_non_cached_pre_activation_fmaps # Used by _get_feature_maps_not_cached
        elif mode in post_activation_methods:
            self.cached_teacher_maps = cached_post_activation_fmaps
            self.non_cached_feature_map_getter = self._get_non_cached_post_activation_fmaps
        else:
            raise ValueError(f'Invalid mode passed, must be in {"/".join(pre_activation_methods + post_activation_methods)}')

        if mode == 'impl':
            self.at_map_generator = self._generate_attention_maps_flattened
            self.loss_computer = self._compute_at_loss
        elif mode == 'paper':
            self.at_map_generator = self._generate_attention_maps_flattened_paper
            self.loss_computer = self._compute_at_loss_paper
        elif mode == 'zoo':
            self.at_map_generator = self._generate_attention_map_zoo
            self.loss_computer = self._compute_at_loss_zoo
        elif mode == 'paper_strict':
            raise NotImplementedError("Paper strict mode not implemented yet")
        else:
            raise ValueError(f'Invalid mode passed, must be in {"/".join(pre_activation_methods + post_activation_methods)}')

        if use_post_activation is not None:
            if use_post_activation is True:
                self.cached_teacher_maps = cached_post_activation_fmaps
                self.non_cached_feature_map_getter = self._get_non_cached_post_activation_fmaps
            elif use_post_activation is False:
                self.cached_teacher_maps = cached_pre_activation_fmaps
                self.non_cached_feature_map_getter = self._get_non_cached_pre_activation_fmaps

        self.get_teacher_feature_maps = self._get_teacher_feature_maps_not_cached
        if self.cached_teacher_maps is not None:
            # Use the function that uses the cached teacher feature maps if cached maps are passed
            self.get_teacher_feature_maps = self._get_teacher_feature_maps_cached
        

    def run_teacher(self):
        return self.cached_teacher_maps is None

    def forward(self, student_logits, teacher_logits, labels, features=None, indices=None):
        """
        Forward pass for computing the attention transfer loss.

        Args:
        - student_logits (torch.Tensor): Logits from the student model.
        - labels (torch.Tensor): Ground truth labels.
        - features (torch.Tensor): Input features/images.
        - indices (list): Indices of the batch features in the context of the entire dataset

        Returns:
        - loss (torch.Tensor): Combined standard and attention transfer loss.
        """
        if features is None:
            raise ValueError("Features are not provided")
        if indices is None:
            raise ValueError("Indices are not provided")
        
        # Get the teacher feature maps
        teacher_feature_maps = self.get_teacher_feature_maps(indices=indices, features=features)
        # Get the student feature maps
        student_feature_maps = self.non_cached_feature_map_getter(self.student, detached=False)
        
        # Create the feature map pairs
        feature_map_pairs = list(zip(student_feature_maps, teacher_feature_maps))

        at_loss = 0
        for student_feature_map, teacher_feature_map in feature_map_pairs:
            loss_component = self.loss_computer(student_feature_map, teacher_feature_map)
            if self.mode in ['paper']:
                loss_component = loss_component / student_feature_map.shape[0] * self.beta
            if self.mode in ['impl', 'zoo', 'paper_strict']:
                loss_component = loss_component * self.beta
                
            at_loss += loss_component
            
        if self.mode == 'zoo':
            at_loss /= len(feature_map_pairs)

        return at_loss
    

    def _generate_attention_maps_flattened(self, feature_map, eps=1e-6):
        return F.normalize(feature_map.pow(2).mean(1).view(feature_map.size(0), -1), eps=eps) # Hobbit implementation method
        # return F.normalize(feature_map.pow(2).mean(1).flatten(1), eps=eps)
    

    def _generate_attention_maps_flattened_paper(self, feature_map, eps=1e-6):
        return F.normalize(feature_map.pow(2).sum(1).flatten(1), eps=eps)
    

    def _generate_attention_map_zoo(self, feature_map, eps=1e-6):
        # am = torch.pow(torch.abs(fm), self.p)
		# am = torch.sum(am, dim=1, keepdim=True)
		# norm = torch.norm(am, dim=(2,3), keepdim=True)
		# am = torch.div(am, norm+eps)
        am = torch.pow(torch.abs(feature_map), 2)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        return torch.div(am, norm+eps)


    def _compute_at_loss(self, student_feature_map, teacher_feature_map):
        at_student = self._generate_attention_maps_flattened(student_feature_map)
        at_teacher = self._generate_attention_maps_flattened(teacher_feature_map)
        return (at_student - at_teacher).pow(2).mean()
    

    def _compute_at_loss_paper(self, student_feature_map, teacher_feature_map):
        at_student = self._generate_attention_maps_flattened_paper(student_feature_map)
        at_teacher = self._generate_attention_maps_flattened_paper(teacher_feature_map)
        return torch.norm(at_student - at_teacher, dim=1).sum()


    def _compute_at_loss_paper_normalized_maps(self, student_feature_map, teacher_feature_map, eps=1e-6):
        at_student = self._generate_attention_maps_flattened_paper(student_feature_map)
        at_teacher = self._generate_attention_maps_flattened_paper(teacher_feature_map)
        # normalise attention maps
        at_student = F.normalize(at_student, dim=1, eps=eps)
        at_teacher = F.normalize(at_teacher, dim=1, eps=eps)
        return torch.norm(at_student - at_teacher, dim=1).sum()


    def _compute_at_loss_zoo(self, student_feature_map, teacher_feature_map):
        at_student = self._generate_attention_map_zoo(student_feature_map)
        at_teacher = self._generate_attention_map_zoo(teacher_feature_map)
        return F.mse_loss(at_student, at_teacher)


    def _get_teacher_feature_maps_cached(self, indices, features):
        teacher_feature_maps = [self.cached_teacher_maps[i] for i in indices]
        tfmap_batch_groups = list(zip(*teacher_feature_maps))
        teacher_feature_maps = [torch.stack(group).to(self.device) for group in tfmap_batch_groups]
        return teacher_feature_maps


    def _get_teacher_feature_maps_not_cached(self, indices, features):
        # self.teacher.generate_logits(features)
        return self.non_cached_feature_map_getter(self.teacher)


    def _get_non_cached_pre_activation_fmaps(self, model: BaseModel, detached=True):
        return model.get_pre_activation_fmaps(detached=detached)


    def _get_non_cached_post_activation_fmaps(self, model: BaseModel, detached=True):
        return model.get_post_activation_fmaps(detached=detached)


    @staticmethod
    def get_group_boundary_indices(model_type):
        if model_type == 'resnet20':
            return [6, 12, 18]
        if model_type == 'resnet32':
            return [10, 20, 30]
        if model_type == 'resnet56':
            return [18, 36, 54]
        if model_type == 'resnet110':
            return [36, 72, 108]
        
        

