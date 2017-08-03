import re
import torch
from torch.autograd import Function
from .utils import FunctionBackend
from .._ext import liborn


class ORAlign1d(Function):

    def __init__(self, nOrientation, return_direction=False):
        super(ORAlign1d, self).__init__()
        self.backend = FunctionBackend(liborn)
        self.nOrientation = nOrientation
        self.return_direction = return_direction

    def forward(self, input):
        mainDirection, output = input.new().byte(), input.new()
        self.backend.set_type(input.type())
        self.backend.RIE_AlignFeature(
            input, 
            mainDirection, 
            output, 
            self.nOrientation)

        if self.return_direction:
            self.save_for_backward(input, mainDirection)
            self.mark_non_differentiable(mainDirection)
            return output, mainDirection
        else:
            self.save_for_backward(input)
            self.mainDirection = mainDirection
            return output

    def backward(self, grad_output):
        if self.return_direction:
            input, mainDirection = self.saved_tensors
        else:
            input, = self.saved_tensors
            mainDirection = self.mainDirection

        grad_input = input.new()
        self.backend.RIE_UnAlignFeature(
            grad_input, 
            mainDirection, 
            grad_output,  
            self.nOrientation)
        return grad_input