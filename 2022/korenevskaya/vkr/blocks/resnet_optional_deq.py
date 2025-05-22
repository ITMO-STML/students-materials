from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Type, List, Optional

import torch
from torch import nn

from .res_blocks.res_block_deq import ResNetLayer, DEQFixedPoint
from .res_blocks.res_block_basic import ResNetBasicBlock
from deq import anderson, broyden


ResNetBlock = Union[Type[ResNetBasicBlock], Type[DEQFixedPoint]]

@dataclass(frozen=True)
class Settings:
    in_planes: int
    conv_1_features: int
    layers_planes: List[int]

@dataclass(frozen=True)
class ResNetArchitecture:
    block: ResNetBlock
    layers: List[int]

CHANNELS_96 = Settings(
    in_planes=96,
    conv_1_features=96,
    layers_planes=[96, 96, 96, 96]
)

RESNET_ARCHITECTURES = {
    'channels_96': ResNetArchitecture(block=ResNetBasicBlock, layers=[3, 4, 6, 3]), 
    'deq_channels_96': ResNetArchitecture(block=DEQFixedPoint, layers=[3, 4, 6, 3]), 
    }

class ResNetModel:

    @staticmethod
    def factory(res_net_size: Union[int, str],
                block_type: Optional[str] = None,
                drop_path_rate: float = 0.0,
                solver=anderson) -> _ResNetModel:


        architecture = RESNET_ARCHITECTURES[res_net_size]

        if block_type == 'basic':
            block = ResNetBasicBlock
        elif block_type == 'deq':
            block = ResNetLayer
        else:
            block = architecture.block

        if res_net_size == 'channels_96':
            return _ResNetModel(block, architecture.layers, settings=CHANNELS_96,
                                activation=nn.ReLU, drop_path_rate=drop_path_rate)
        elif res_net_size == 'deq_channels_96':
            return _ResNetModel(block, architecture.layers, settings=CHANNELS_96, solver=solver,
                                activation=nn.ReLU, drop_path_rate=drop_path_rate, deq=True)
        else:
            raise NotImplementedError


class _ResNetModel(nn.Module):

    def __init__(
            self,
            block: ResNetBlock,
            layers: List[int],
            settings: Settings,
            solver=anderson,
            m=5, 
            max_iter=50, 
            activation: Type[nn.Module] = nn.ReLU,
            b_solver=anderson,
            drop_path_rate: float = 0.0,
            deq=False
    ) -> None:
        super(_ResNetModel, self).__init__()

        self.in_planes = settings.in_planes

        self.conv1 = self._make_initial_conv()
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = activation(inplace=True)
        drop_path_rate = drop_path_rate
        self.net_num_blocks = sum(layers)
        self.layer1 = self._make_pooling_layer(block, solver, 
                                       m, max_iter, settings.layers_planes[0], layers[0], stride=1, activation=activation, b_solver=b_solver, deq=deq)
        self.layer2 = self._make_pooling_layer(block, solver,
                                       m, max_iter, settings.layers_planes[1], layers[1], stride=2, activation=activation, b_solver=b_solver, deq=deq)
        self.layer3 = self._make_pooling_layer(block, solver,
                                       m, max_iter, settings.layers_planes[2], layers[2], stride=2, activation=activation, b_solver=b_solver, deq=deq)
        self.layer4 = self._make_pooling_layer(block, solver,
                                       m, max_iter, settings.layers_planes[3], layers[3], stride=2, activation=activation, b_solver=b_solver, deq=deq)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, z=1) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        def seq_forward(x, layer_n):
            for f in layer_n:
                x = f(x)
            return x         
        x = seq_forward(x, self.layer1)
        x = seq_forward(x, self.layer2)
        x = seq_forward(x, self.layer3)
        x = seq_forward(x, self.layer4)

        return x

    def _make_initial_conv(self):
        return nn.Conv2d(1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_pooling_layer(
        self,
        block: ResNetBlock,
        solver, 
        m, 
        max_iter, 
        planes: int,
        blocks: int,
        stride: int = 1,
        activation: Type[nn.Module] = nn.ReLU,
        b_solver=anderson, 
        deq=False
    ) -> nn.Module:

        down_sample = None

        layers = list()
        if deq:
            layers.append(block(ResNetLayer(self.in_planes, planes, 1, down_sample, activation=activation), solver, m, max_iter, b_solver=b_solver))
            self.in_planes = planes * block.expansion
            if stride != 1:
                layers.append(torch.nn.AvgPool2d(2, stride=(2,2), padding=(0,0))) 
            for _ in range(blocks):
                layers.append(block(ResNetLayer(self.in_planes, planes, activation=activation), solver, m, max_iter))
    
        else:    
            layers.append(block(self.in_planes, planes, 1, down_sample, activation=activation))
            self.in_planes = planes * block.expansion
            if stride != 1:
                layers.append(torch.nn.AvgPool2d(2, stride=(2,2), padding=(0,0))) 
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, planes, activation=activation))

        return nn.Sequential(*layers)  
