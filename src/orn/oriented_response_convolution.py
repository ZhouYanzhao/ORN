import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from orn.active_rotating_filters import mapping_rotate


class ORConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, arf_config, kernel_size, 
        stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ORConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_orientation, self.num_rotation = _pair(arf_config)
        self.kernel_size = _pair(kernel_size)
        if (self.kernel_size[0] not in (1, 3)) or (self.kernel_size[1] not in (1, 3)):
            self.kernel_size = _pair(3)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert (math.log(self.num_orientation) + 1e-5) % math.log(2) < 1e-3, 'invalid num_orientation {}'.format(self.num_orientation)
        assert (math.log(self.num_rotation) + 1e-5) % math.log(2) < 1e-3, 'invalid num_rotation {}'.format(self.num_rotation)
        
        self.register_buffer("indices", self.get_indices())
        self.register_parameter('weight', nn.Parameter(torch.empty(out_channels, in_channels, self.num_orientation, *self.kernel_size)))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.empty(out_channels * self.num_rotation)))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.num_orientation
        for k in self.kernel_size:
            n *= k
        with torch.no_grad():
            self.weight.normal_(0, math.sqrt(2.0 / n))
            if self.bias is not None:
                self.bias.zero_()

    def get_indices(self):
        kernel_indices = {
            1: {
                0: (1,),
                45: (1,),
                90: (1,),
                135: (1,),
                180: (1,),
                225: (1,),
                270: (1,),
                315: (1,)
            },
            3: {
                0: (1,2,3,4,5,6,7,8,9),
                45: (2,3,6,1,5,9,4,7,8),
                90: (3,6,9,2,5,8,1,4,7),
                135: (6,9,8,3,5,7,2,1,4),
                180: (9,8,7,6,5,4,3,2,1),
                225: (8,7,4,9,5,1,6,3,2),
                270: (7,4,1,8,5,2,9,6,3),
                315: (4,1,2,7,5,3,8,9,6)
            }
        }
        delta_orientation = 360 / self.num_orientation
        delta_rotation = 360 / self.num_rotation
        kH, kW = self.kernel_size
        indices = torch.ByteTensor(self.num_orientation * kH * kW, self.num_rotation)
        for i in range(0, self.num_orientation):
            for j in range(0, kH * kW):
                for k in range(0, self.num_rotation):
                    angle = delta_rotation * k
                    layer = (i + math.floor(angle / delta_orientation)) % self.num_orientation
                    kernel = kernel_indices[kW][angle][j]
                    indices[i * kH * kW + j, k] = int(layer * kH * kW + kernel)
        return indices.view(self.num_orientation, kH, kW, self.num_rotation)

    def rotate_arf(self):
        return mapping_rotate(self.weight, self.indices)

    def forward(self, input):
        arf_weight = self.rotate_arf()
        return F.conv2d(input, arf_weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

    def __repr__(self):
        arf_config = f'[{self.num_orientation}]' if self.num_orientation == self.num_rotation else f'[{self.num_orientation}-{self.num_rotation}]'
        s = f'{self.__class__.__name__}({arf_config} {self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}'
        if self.padding != (0,) * len(self.padding):
            s += f', padding={self.padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        s += f', bias={bool(self.bias is not None)}'
        s += ')'
        return s
