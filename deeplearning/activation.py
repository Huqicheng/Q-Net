from typing import Dict,Callable
from deeplearning.layers import Layer
from deeplearning.tensor import Tensor
import numpy as np
from deeplearning.func import *

# this is the Function type for Activation class
F = Callable[[Tensor],Tensor]


class Activation(Layer):

    def __init__(self,
                 name,
                 f: F,
                 f_prime: F) -> None:
        super().__init__(name)
        self.f = f
        self.f_prime = f_prime
    
    
    def forward(self,inputs: Tensor, **kwargs) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    '''
        y = g(x) z = f(z)
        dy/dz = g'(x) * f'(z)
    '''
    def backward(self,grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad



class Softmax(Activation):
    def __init__(self, name) -> None:
        super().__init__(name, softmax, softmax_derivative)



class Tanh(Activation):
    def __init__(self, name) -> None:
        super().__init__(name, tanh, tanh_derivative)



class Sigmoid(Activation):
    def __init__(self, name) -> None:
        super().__init__(name, sigmoid, sigmoid_derivative)



class ReLU(Activation):
    def __init__(self, name) -> None:
        super().__init__(name, relu, relu_derivative)
