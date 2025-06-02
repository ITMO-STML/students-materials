from torch.nn import Parameter
import torch
from torch import nn
import math


class CurricularAAM(nn.Module):
    def __init__(self, in_features, out_features, sub_centers=1,
                 m=0.35, s=32., fixed=False):
        super(CurricularAAM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sub_centers = sub_centers
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features*sub_centers))
        self.fixed = fixed
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)
        # self.kernel_norm = torch.nn.functional.normalize(self.kernel, dim=0)

    # TODO: Split curricular and AAD logic
    # TODO: Unify with CosLinear
    def forward(self, embbedings,  label=None,
                add_labels=None,
                lambda_f=None):
        # embbedings = l2_norm(embbedings, axis=1)
        # kernel_norm = l2_norm(self.kernel, axis=0)
        embbedings_norm = torch.nn.functional.normalize(embbedings, dim=1)
        kernel_norm = torch.nn.functional.normalize(self.kernel, dim=0)

        cos_theta = torch.mm(embbedings_norm, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        if self.sub_centers > 1:
            cos_theta = cos_theta.reshape(-1, self.out_features, self.sub_centers)
            cos_theta, _ = cos_theta.max(axis=-1)
        if self.training:
            with torch.no_grad():
                origin_cos = cos_theta.clone()
            target_logit = cos_theta[torch.arange(0, embbedings_norm.size(0)), label].view(-1, 1)
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            cos_theta_m = cos_theta_m.to(target_logit.dtype)
            mask = cos_theta > cos_theta_m
            final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

            hard_example = cos_theta[mask]
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            cos_theta[mask] = hard_example * (self.t.to(hard_example.dtype) + hard_example)
            cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
            output = cos_theta * self.s
            return output, origin_cos * self.s
        else:
            return cos_theta
