import torch
from torch.autograd import Function
from .utils import FunctionBackend
from .._ext import liborn


class MappingRotate(Function):

    def __init__(self, indices):
        super(MappingRotate, self).__init__()
        self.backend = FunctionBackend(liborn)
        self.indices = indices.byte()

    def forward(self, input):
        output = input.new()
        self.backend.set_type(input.type())
        self.backend.ARF_MappingRotate(input, self.indices, output)
        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = input.new()
        self.backend.ARF_MappingAlign(grad_input, self.indices, grad_output)
        return grad_input