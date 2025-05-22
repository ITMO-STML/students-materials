import torch
from torch import nn
from timm.models.layers import DropBlock2d, DropPath
from deq import anderson, broyden, anderson3stable
import torch.nn.functional as F
import torch.autograd as autograd

def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

class ResNetLayer(nn.Module):
    def __init__(self, in_planes, planes, stride=1,
                 down_sample=None, norm_layer=None,
                 activation=nn.ReLU,
                 ):

        super(ResNetLayer, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = activation(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.down_sample = down_sample
        self.stride = stride
        self.drop_block = None

    def forward(self, x, z):
        y = self.relu(self.bn1(self.conv1(z)))
        return self.relu(z + self.bn2(x + self.conv2(y)))

class DEQFixedPoint(nn.Module):
    expansion = 1

    def __init__(self, f, solver, m=5, max_iter=1000, b_solver=anderson, **kwargs):
        super().__init__()
        self.f = f
        self.m = m
        self.max_iter = max_iter
        self.solver = solver
        self.b_solver = b_solver
        self.kwargs = kwargs
        
    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            if self.solver==broyden:
                z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), threshold=20, **self.kwargs)
            elif self.solver==anderson:
                z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), self.m, self.max_iter, **self.kwargs)
            else: 
                z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), threshold=30, **self.kwargs)
        z = self.f(z,x)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            if self.b_solver==broyden:
                g, self.backward_res = self.b_solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               torch.zeros_like(grad), threshold=20, **self.kwargs)
            else:
                g, self.backward_res = self.b_solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               torch.zeros_like(grad), self.m, self.max_iter, **self.kwargs)
            return g
                
        if z.requires_grad:
            z.register_hook(backward_hook)
        return z    