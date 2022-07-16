import os

import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load


LIB_BACKEND = load('rotation_invariant_encoding', [os.path.join(os.path.dirname(os.path.abspath(__file__)), v) for v in [
        'lib/rotation_invariant_encoding.cpp', 
        'lib/rotation_invariant_encoding.cu'
    ]], verbose=True)

class AlignFeature(Function):

    @staticmethod
    def forward(ctx, input, num_orientation):
        output, mainDirection = LIB_BACKEND.align_feature(input, num_orientation)
        ctx.num_orientation = num_orientation
        ctx.mainDirection = mainDirection
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = LIB_BACKEND.unalign_feature(grad_output.contiguous(), ctx.mainDirection, ctx.num_orientation)
        return grad_input, None

def oralign1d(input, num_orientation):
  return AlignFeature.apply(input, num_orientation)

class ORAlign1d(nn.Module):

    def __init__(self, num_orientation):
        super(ORAlign1d, self).__init__()
        self.num_orientation = num_orientation

    def forward(self, input):
        if input.ndim == 2:
            num_batch, num_feature = input.shape
            return oralign1d(input.view(num_batch, num_feature, 1, 1), self.num_orientation)[:, :, 0, 0]
        elif input.ndim == 4:
            return oralign1d(input, self.num_orientation)
        else:
            raise NotImplementedError(f'Dimension {input.ndim} is not supported.')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.num_orientation})'

def orpool1d(input, num_orientation):
    num_batch, num_channel, kH, kW = input.shape
    num_feature = num_channel // num_orientation
    return input.view(num_batch, num_feature, num_orientation, kH, kW).max(2).values

class ORPool1d(nn.Module):

    def __init__(self, num_orientation):
        super(ORPool1d, self).__init__()
        self.num_orientation = num_orientation

    def forward(self, input):
        if input.ndim == 2:
            num_batch, num_feature = input.shape
            return orpool1d(input.view(num_batch, num_feature, 1, 1), self.num_orientation)[:, :, 0, 0]
        elif input.ndim == 4:
            return orpool1d(input, self.num_orientation)
        else:
            raise NotImplementedError(f'Dimension {input.ndim} is not supported.')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.num_orientation})'
