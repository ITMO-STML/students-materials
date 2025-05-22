from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.functional import relu

from enum import Enum


class StatPoolMode(Enum):
    M = 0
    V = 1
    MV = 2


class StatPoolLayer(nn.Module):

    @classmethod
    def by_string_mode(cls, mode: str) -> StatPoolLayer:
        mode = StatPoolMode[mode]
        return cls(mode)

    """ Building of SGPK (Snyder, Garcia-Romero, Povey, Khudanpur) statistics pooling layer's model. Implementation
    is based on work Snyder D. et al. X-vectors: Robust DNN embeddings for speaker recognition // ICASSP, Calgary,
    Canada. – 2018
    """
    def __init__(self, stat_pool_mode: StatPoolMode, dim=-1):
        super(StatPoolLayer, self).__init__()

        ################################################################################################################
        #
        # mode:      0 - average is calculated only;
        #            1 - standard deviation is calculated only,
        #            2 - average and standard deviation is calculated
        #
        ################################################################################################################
        self.mode = stat_pool_mode
        self.dim = dim

    def forward(self, x, **kwargs):
        mean_x = x.mean(self.dim)
        mean_x2 = x.pow(2).mean(self.dim)

        std_x = relu(mean_x2 - mean_x.pow(2)).sqrt()
        
        if self.mode == StatPoolMode.M:
            out = mean_x
        elif self.mode == StatPoolMode.V:
            out = std_x
        elif self.mode == StatPoolMode.MV:
            out = torch.cat([mean_x, std_x], dim=-1)
        else:
            raise ValueError('Operation\'s mode is incorrect')
        out = torch.flatten(out, 1)
        return out
