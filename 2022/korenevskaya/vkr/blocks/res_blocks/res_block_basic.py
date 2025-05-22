from torch import nn
from timm.models.layers import DropBlock2d


def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


class ResNetBasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 down_sample=None, norm_layer=None,
                 activation=nn.ReLU,
                 drop_block_prob=0.0, drop_block_size=5,
                 drop_block_gamma_scale=0.25,
                 drop_path=None):

        super(ResNetBasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = activation(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.down_sample = down_sample
        self.stride = stride
        self.drop_path = drop_path
        if drop_block_prob > 0.0:
            self.drop_block = DropBlock2d(drop_block_prob,
                                          drop_block_size,
                                          drop_block_gamma_scale)
        else:
            self.drop_block = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.drop_block is not None:
            x = self.drop_block(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.drop_block is not None:
            x = self.drop_block(x)
        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


