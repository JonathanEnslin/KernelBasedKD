import torch.nn.functional as F
from models.base_model import BaseModel
from loss_functions.base_loss_fn import BaseLoss

'''
Adapted from:
[1] https://github.com/HobbitLong/RepDistiller/tree/master
and
[2] https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ft.py
and
[3] https://github.com/yoshitomo-matsubara/torchdistill/blob/db102321a0c7fbe94b88465332dd99e8f10d18e5/torchdistill/losses/mid_level.py#L315
'''


class FTLoss(BaseLoss):
    def __init__(self, translator, paraphraser, student, teacher, p=1):
        super(FTLoss, self).__init__()
        self.p = p
        self.student = student
        self.teacher = teacher
        self.translator = translator
        self.paraphraser = paraphraser


    def get_factors(self):
        # get the student and teacher feature maps
        student_fmap = self.student.get_post_activation_fmaps(detached=False)[-1] # get the last layer feature map
        teacher_fmap = self.teacher.get_post_activation_fmaps(detached=True)[-1] # get the last layer feature map
        student_fmaps = [student_fmap]
        teacher_fmaps = [teacher_fmap]
        student_factors = []
        teacher_factors = []
        for student_fmap, teacher_fmap in zip(student_fmaps, teacher_fmaps):
            student_fmap = self.translator(student_fmap)
            teacher_fmap = self.paraphraser(teacher_fmap, is_factor=True)
            student_factors.append(self.generate_factor(student_fmap))
            teacher_factors.append(self.generate_factor(teacher_fmap))
            # check that dims are the same
            assert student_factors[-1].shape == teacher_factors[-1].shape
        return student_factors, teacher_factors

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
        student_factors, teacher_factors = self.get_factors()

        loss = 0
        for student_factor, teacher_factor in zip(student_factors, teacher_factors):
            loss += self.calc_loss(student_factor, teacher_factor)

        return loss


