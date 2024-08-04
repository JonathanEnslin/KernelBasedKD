import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import models.resnet

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


class ATLoss(nn.Module):
    def __init__(self, student, teacher, beta=1e3, layer_index_pairs=None, mode='impl', scaler='map_size'):
        """
        Initializes the VanillaKDLoss module.
        
        Args:
        - alpha (float): Weight for the distillation loss.
        - temperature (float): Temperature for the distillation process.
        - teacher (nn.Module): Pre-trained teacher model (optional).
        """
        super(ATLoss, self).__init__()
        self.student = student
        self.teacher = teacher
        self.beta = beta
        self.layer_index_pairs = layer_index_pairs
        self.standard_criterion = nn.CrossEntropyLoss()
        self.mode = mode

        if mode == 'impl':
            self.at_map_generator = self._generate_attention_maps_flattened_paper
            self.loss_computer = self._compute_at_loss_paper
        elif mode == 'paper':
            self.at_map_generator = self._generate_attention_maps_flattened
            self.loss_computer = self._compute_at_loss
        elif mode == 'zoo':
            self.at_map_generator = self._generate_attention_map_zoo
            self.loss_computer = self._compute_at_loss_zoo
        elif mode == 'paper_strict':
            self.at_map_generator = self._generate_attention_maps_flattened_paper
            self.loss_computer = self._compute_at_loss_paper_normalized_maps
        else:
            raise ValueError("Invalid mode, must be in ['impl', 'paper', 'zoo', 'paper_strict']")
        

    def forward(self, student_logits, labels, features=None, indices=None):
        """
        Forward pass for computing the KD loss.

        Args:
        - student_logits (torch.Tensor): Logits from the student model.
        - labels (torch.Tensor): Ground truth labels.
        - teacher_logits (torch.Tensor): Logits from the teacher model (optional).
        - features (torch.Tensor): Input features/images (optional, if teacher model is provided).

        Returns:
        - loss (torch.Tensor): Combined KD and cross-entropy loss.
        """
        if features is None:
            raise ValueError("Features are not provided")
        self.teacher.generate_logits(features)

        teacher_feature_maps = self.teacher.get_feature_maps()    
        student_feature_maps = self.student.get_feature_maps()

        feature_map_pairs = [(student_feature_maps[i], teacher_feature_maps[j]) for i, j in self.layer_index_pairs]

        if self.mode == 'zoo':
            # use the post activations
            teacher_feature_maps = self.teacher.layer_group_output_feature_maps
            student_feature_maps = self.student.layer_group_output_feature_maps
            feature_map_pairs = list(zip(student_feature_maps, teacher_feature_maps))


        at_loss = 0
        for student_feature_map, teacher_feature_map in feature_map_pairs:
            loss_component = self.loss_computer(student_feature_map, teacher_feature_map)
            if self.mode == 'paper':
                loss_component = loss_component / student_feature_map.shape[0] * self.beta
            elif self.mode == 'impl':
                loss_component = loss_component * self.beta
            elif self.mode == 'zoo':
                loss_component = loss_component * self.beta
            elif self.mode == 'paper_strict':
                loss_component = loss_component * self.beta
                
            at_loss += loss_component
            
        if self.mode == 'zoo':
            at_loss /= len(feature_map_pairs)

        student_loss = self.standard_criterion(student_logits, labels)
        return student_loss + at_loss
    

    def _generate_attention_maps_flattened(self, feature_map):
        return F.normalize(feature_map.pow(2).mean(1).flatten(1))
    

    def _generate_attention_maps_flattened_paper(self, feature_map):
        return F.normalize(feature_map.pow(2).sum(1).flatten(1))
    

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

    def _compute_at_loss_paper_normalized_maps(self, student_feature_map, teacher_feature_map):
        at_student = self._generate_attention_maps_flattened_paper(student_feature_map)
        at_teacher = self._generate_attention_maps_flattened_paper(teacher_feature_map)
        # normalise attention maps
        at_student = F.normalize(at_student, dim=1)
        at_teacher = F.normalize(at_teacher, dim=1)
        return torch.norm(at_student - at_teacher, dim=1).sum()


    def _compute_at_loss_zoo(self, student_feature_map, teacher_feature_map):
        at_student = self._generate_attention_map_zoo(student_feature_map)
        at_teacher = self._generate_attention_map_zoo(teacher_feature_map)
        return F.mse_loss(at_student, at_teacher)

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

