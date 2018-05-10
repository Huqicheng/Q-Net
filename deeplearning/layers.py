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
        https://www.quora.com/How-does-batch-normalization-behave-differently-at-training-time-and-test-time
    """
    def __init__(self,
                 name,
                 input_size: int,
                 decay: float = 0.99) -> None:
        super().__init__(name)
        self.params['gamma'] = np.ones((input_size,))
        self.params['beta'] = np.zeros((input_size,))
        self.global_mean = np.zeros((input_size,))
        self.global_var = np.zeros((input_size,))
        self.decay = decay
        self.input_size = input_size
        self.eps = np.finfo(float).eps

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        training = kwargs["training"]
        
        if training == False:
            x_hat = (inputs - self.global_mean) / np.sqrt(self.global_var + self.eps)
            gammax = self.params['gamma'] * x_hat
            self.out = gammax + self.params['beta']
            return self.out
        
        N,D = inputs.shape
        mu = 1./N * np.sum(inputs,axis=0)
        self.xmu = inputs - mu
        sq = self.xmu ** 2
        self.var = 1./N * np.sum(sq, axis=0)
        self.sqrtvar = np.sqrt(self.var+self.eps)
        self.ivar = 1./self.sqrtvar
        self.xhat = self.xmu * self.ivar
        gammax = self.params['gamma'] * self.xhat
        self.out = gammax + self.params['beta']
        
        self.global_mean = self.global_mean * self.decay + mu * (1 - self.decay)
        self.global_var = self.global_var * self.decay + self.var * (1 - self.decay)
        
        return self.out

    def backward(self, grad: Tensor) -> Tensor:
        N,D = grad.shape
        dbeta = np.sum(grad, axis=0)
        dgammax = grad
        dgamma = np.sum(dgammax*self.xhat, axis=0)
        dxhat = dgammax * self.params['gamma']
        divar = np.sum(dxhat*self.xmu, axis=0)
        dxmu1 = dxhat * self.ivar
        dsqrtvar = -1./(self.sqrtvar**2) * divar
        dvar = 0.5 * 1./np.sqrt(self.var+self.eps) * dsqrtvar
        dsq = 1./N * np.ones((N,D)) * dvar
        dxmu2 = 2 * self.xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        dx2 = 1. /N * np.ones((N,D)) * dmu
        dx = dx1 + dx2
        self.grads['gamma'] = dgamma
        self.grads['beta'] = dbeta
        return dx

#    def backward(self, dy: Tensor) -> Tensor:
#        N,D = dy.shape
#        h = self.out
#        mu = 1./N*np.sum(h, axis = 0)
#        var = 1./N*np.sum((h-mu)**2, axis = 0)
#        self.grads['beta'] = np.sum(dy, axis=0)
#        self.grads['gamma'] = np.sum((h - mu) * (var + np.finfo(float).eps)**(-1. / 2.) * dy, axis=0)
#        dh = (1. / N) * self.params['gamma'] * (var + np.finfo(float).eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0) - (h - mu) * (var + np.finfo(float).eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))
#        return dh






