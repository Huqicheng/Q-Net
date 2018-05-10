from deeplearning.tensor import Tensor

import numpy as np
from typing import Dict

class Layer:

    def __init__(self, name) -> None:
        self.params: Dict[Str,Tensor] = {}
        self.grads: Dict[Str,Tensor] = {}
        self.name = name

    def forward(self,inputs: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def backward(self,grad: Tensor) -> Tensor:
        raise NotImplementedError



class Dense(Layer):

    '''
    inputs@w+b
        
    inputs = (batch_size,input_size)
    outputs = (batch_size,output_size)
    '''
    def __init__(self,
                 name,
                 input_size: int,
                 output_size: int) -> None:
        super().__init__(name)
        self.params["w"] = np.random.randn(input_size,output_size)
        self.params["b"] = np.random.randn(output_size)


    def forward(self,inputs: Tensor, **kwargs) -> Tensor:
        self.inputs = inputs
        return inputs@self.params["w"]+self.params["b"]
    
    '''
        d net / d w
        d net / d b which is not a nueron in the nn, so doesn't backprop it
        
        d (f(wx+b)) / d x = f'(wx+b)*(d (x@w+b) / d x) = grad * wT
    '''
    def backward(self,grad: Tensor) -> Tensor:
        self.grads["b"] = np.sum(grad,axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T




class Dropout(Layer):
    '''
        randomly ignore some signals
        reference:
        https://blog.csdn.net/stdcoutzyx/article/details/49022443
        https://blog.csdn.net/hjimce/article/details/50413257
    '''
    def __init__(self,
                 name,
                 dropout_rate=0.5) -> None:
        super().__init__(name)
        self.dropout_rate = dropout_rate

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        is_training = kwargs['training']
        if is_training:
            self.mask = np.random.binomial(n=1,p=1-self.dropout_rate,size=inputs.shape[1])
        else:
            self.mask = 1
        return inputs*self.mask/(1-self.dropout_rate)

    def backward(self, grad: Tensor) -> Tensor:
        return grad*self.mask/(1-self.dropout_rate)



class BatchNormalization(Layer):
    """
        references:
        https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    """





