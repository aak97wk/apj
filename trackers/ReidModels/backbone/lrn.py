
import jittor as jt
from jittor import init
from jittor import nn

class SpatialCrossMapLRNFunc(Function):

    def __init__(self, size, alpha=0.0001, beta=0.75, k=1):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def execute(self, input):
        self.save_for_backward(input)
        self.lrn = SpatialCrossMapLRNOld(self.size, self.alpha, self.beta, self.k)
        self.lrn.astype(input.dtype)
        return self.lrn.forward(input)

    def backward(self, grad_output):
        (input,) = self.saved_tensors
        return self.lrn.backward(input, grad_output)

class SpatialCrossMapLRN(Module):

    def __init__(self, size, alpha=0.0001, beta=0.75, k=1):
        super(SpatialCrossMapLRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def execute(self, input):
        return SpatialCrossMapLRNFunc(self.size, self.alpha, self.beta, self.k)(input)
