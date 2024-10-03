import torch.nn.functional as F
from models.base_model import BaseModel
from models.resnet import ResNet
from loss_functions.base_loss_fn import BaseLoss

'''
Adapted from:
[1] https://github.com/HobbitLong/RepDistiller/tree/master
and
[2] https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ft.py
and
[3] https://github.com/yoshitomo-matsubara/torchdistill/blob/db102321a0c7fbe94b88465332dd99e8f10d18e5/torchdistill/losses/mid_level.py#L315
'''


class KFTLoss(BaseLoss):
    def __init__(self, translator, paraphraser, student, teacher, group, p=1):
        super(KFTLoss, self).__init__()
        self.p = p
        self.student: BaseModel = student
        self.teacher: BaseModel = teacher
        self.translator = translator
        self.paraphraser = paraphraser
        self.group = group


    # This method treats channels of the kernels as batches
    def get_kfactors(self):
        student_kernels, teacher_kernels = self.get_kernels()

        student_factors = []
        teacher_factors = []
        for stu_kernel, tea_kernel in zip(student_kernels, teacher_kernels):
            student_factor = self.translator(stu_kernel)
            teacher_factor = self.paraphraser(tea_kernel, is_factor=True)
            student_factors.append(self.generate_factor(student_factor))
            teacher_factors.append(self.generate_factor(teacher_factor))
            # check that dims are the same
            assert student_factors[-1].shape == teacher_factors[-1].shape
        return student_factors, teacher_factors


    def get_kernels(self):
        self.student: ResNet = self.student
        self.teacher: ResNet = self.teacher
        # get the last layer weights for now
        if self.group == 'group1':
            student_weights = self.student.get_kernel_weights_subset([self.student.group1indices[-1]], detached=False)
            teacher_weights = self.teacher.get_kernel_weights_subset([self.teacher.group1indices[-1]], detached=True)
        elif self.group == 'group2':
            student_weights = self.student.get_kernel_weights_subset([self.student.group2indices[-1]], detached=False)
            teacher_weights = self.teacher.get_kernel_weights_subset([self.teacher.group2indices[-1]], detached=True)
        elif self.group == 'group3':
            student_weights = self.student.get_kernel_weights_subset([self.student.group3indices[-1]], detached=False)
            teacher_weights = self.teacher.get_kernel_weights_subset([self.teacher.group3indices[-1]], detached=True)
        return student_weights, teacher_weights
    

    def generate_factor(self, feature_map):
        # Feature map shape: [B, C, H, W]
        # Using impl from [1] [3]
        # Flatten among C, H, W
        # shape: [B, C, H, W] -> [B, C*H*W]
        flattened = feature_map.view(feature_map.size(0), -1)
        normalized = F.normalize(flattened, p=2, dim=1, eps=1e-8)
        return normalized


    def calc_loss(self, factor1, factor2):
        # Using impl from [1] [3]
        if self.p == 1:
            return F.l1_loss(factor1, factor2)
        else:
            return F.mse_loss(factor1, factor2)


    def run_teacher(self):
        return True


    def forward(self, student_logits, teacher_logits, labels, features=None, indices=None):
        student_factors, teacher_factors = self.get_kfactors()

        loss = 0
        for student_factor, teacher_factor in zip(student_factors, teacher_factors):
            loss += self.calc_loss(student_factor, teacher_factor)

        return loss


