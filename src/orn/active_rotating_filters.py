import os

from torch.autograd import Function
from torch.utils.cpp_extension import load


LIB_BACKEND = load('active_rotating_filters', [os.path.join(os.path.dirname(os.path.abspath(__file__)), v) for v in [
        'lib/active_rotating_filters.cpp', 
        'lib/active_rotating_filters.cu'
    ]], verbose=True)

class MappingRotate(Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.indices = indices
        output = LIB_BACKEND.mapping_rotate(input, ctx.indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = LIB_BACKEND.mapping_align(grad_output.contiguous(), ctx.indices)
        return grad_input, None

def mapping_rotate(weight, indices):
  return MappingRotate.apply(weight, indices)
