from deeplearning.pool_utils import *
from deeplearning.layers import Layer
from deeplearning.conv_utils import *
from deeplearning.tensor import Tensor
import numpy as np


"""
    Stanford CS231n
    
    http://cs231n.github.io/convolutional-networks/
"""

def _pool_forward(X, pool_fun, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1
    
    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')
    
    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, stride=stride)
    
    out, pool_cache = pool_fun(X_col)
    
    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1)



    cache = (X, size, stride, X_col, pool_cache)
    
    return out, cache


def _pool_backward(dout, dpool_fun, cache):
    X, size, stride, X_col, pool_cache = cache
    n, d, w, h = X.shape
    
    dX_col = np.zeros_like(X_col)
    dout_col = dout.transpose(2, 3, 0, 1).ravel()
    
    dX = dpool_fun(dX_col, dout_col, pool_cache)
    
    dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, stride=stride)
    dX = dX.reshape(X.shape)
    
    return dX


class Pool_2D(Layer):
    
    def __init__(self, name, pool_fun, dpool_fun,
                 size: int = 2,
                 stride: int = 2) -> None:
        
        super().__init__(name)
        self.size = size
        self.stride = stride
        self.pool_fun = pool_fun
        self.dpool_fun = dpool_fun
        
    
    def forward(self,inputs: Tensor, **kwargs) -> Tensor:
        out, self.cache = _pool_forward(inputs, self.pool_fun, self.size, self.stride)
        return out

    def backward(self,grad: Tensor) -> Tensor:
        return _pool_backward(grad, self.dpool_fun, self.cache)


class Max_Pool_2D(Pool_2D):
    def __init__(self,
                 name,
                 size: int = 2,
                 stride: int = 2 ):
        super().__init__(name,maxpool,dmaxpool,size,stride)


class Avg_Pool_2D(Pool_2D):
    def __init__(self,
                 name,
                 size: int = 2,
                 stride: int = 2 ):
        super().__init__(name,avgpool,davgpool,size,stride)






