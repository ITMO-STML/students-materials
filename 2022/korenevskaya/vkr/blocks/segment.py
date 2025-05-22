import torch.nn as nn
from typing import List, Union
import torch


class MaxoutLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return torch.max(self.linear1(x), self.linear2(x))
    

class MaxoutSegmentLevel(nn.Module):

    def __init__(self, input_dim: Union[int, List[int]], output_dim: Union[int, List[int]], enable_batch_norm: bool, fixed: bool = False):
        super(MaxoutSegmentLevel, self).__init__()

        if isinstance(input_dim, int) and isinstance(output_dim, int):
            input_dim = [input_dim]
            output_dim = [output_dim]
        self.num_layers = len(input_dim)
        self.enable_batch_norm = enable_batch_norm
        self.layers = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.fixed = fixed

        for idx in range(self.num_layers):
            self.layers.append(MaxoutLinear(input_dim[idx], output_dim[idx]))

            if self.enable_batch_norm:
                self.bn.append(nn.BatchNorm1d(output_dim[idx], affine=False))

    def forward(self, x, **kwargs):

        for idx in range(self.num_layers):
            x = self.layers[idx](x)

            if self.enable_batch_norm:
                x = self.bn[idx](x)

        return x
