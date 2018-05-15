import numpy as np
from deeplearning.layers import Layer
from deeplearning.conv_utils import *

"""
    Stanford CS239n
    
    https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
"""


def conv_forward(X, W, b, stride=1, padding=1):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    
    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)
    
    out = W_col @ X_col + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)
    
    cache = (X, W, b, stride, padding, X_col)
    
    return out, cache

"""
    1 padding and 1 stride will keep the input size
"""
def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape
    
    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)
    
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ X_col.T
    dW = dW.reshape(W.shape)
    
    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)
    
    return dX, dW, db







class Convolution_2D(Layer):
    
    def __init__(self,
                 name,
                 input_shape: Tuple[int],
                 output_shape: Tuple[int],
                 filter_shape: Tuple[int],
                 padding: int = 1,
                 stride: int = 1) -> None:
        super().__init__(name)
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        
        input_n,input_d,input_h,input_w = input_shape
        fan_out = filter_shape[0] * np.prod(filter_shape[2:]
        W_bound = numpy.sqrt(6. / ( input_n + fan_out))
        self.params["w"] = np.asarray(
                                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                dtype="float64"
                            )
        self.params["b"] = np.zeros((filter_shape[0],), dtype="float64")
            
            
    def forward(self,inputs: Tensor, **kwargs) -> Tensor:
        out,self.cache = conv_forward(
                                    inputs,
                                    self.params['w'],
                                    self.params['b'],
                                    stride=self.stride,
                                    padding=self.padding
                                )
                                            
        return out
            
    
    def backward(self,grad: Tensor) -> Tensor:
        dx, self,grads['w'], self.grads['b'] = conv_backward(grad, self.cache)
        return dx


                                            
