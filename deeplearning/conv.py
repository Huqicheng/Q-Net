import numpy as np
from deeplearning.layers import Layer
from deeplearning.conv_utils import *
from deeplearning.tensor import Tensor
from typing import Tuple

"""
    Stanford CS239n
    
    https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
    
    http://cs231n.github.io/convolutional-networks/
"""

def pad_same(X, filter_h, filter_w, stride):
    """
        new_shape = input_shape/stride
        
        reference:
            https://www.jianshu.com/p/05c4f1621c7e
    """
    
    N,C,H,W = X.shape
    
    w_need_pad = (W/stride - 1) * stride + filter_w - W
    h_need_pad = (H/stride - 1) * stride + filter_h - H
    
    w_need_pad = max(0,int(w_need_pad))
    h_need_pad = max(0,int(h_need_pad))
    
    w_padding = int(w_need_pad / 2)
    h_padding = int(h_need_pad / 2)

    if w_need_pad % 2 == 0:
        if h_need_pad % 2 == 0:
            w_pad_left,w_pad_right,h_pad_up,h_pad_down = w_padding,w_padding,h_padding,h_padding
            x_padded = np.pad(X, ((0,0),(0,0),(h_padding,h_padding),(w_padding,w_padding)), mode="constant")
        else:
            assert h_padding+1+h_padding == h_need_pad
            w_pad_left,w_pad_right,h_pad_up,h_pad_down = w_padding,w_padding,h_padding+1,h_padding
            x_padded = np.pad(X, ((0,0),(0,0),(h_padding+1,h_padding),(w_padding,w_padding)), mode="constant")
    else:
        if h_need_pad % 2 == 0:
            w_pad_left,w_pad_right,h_pad_up,h_pad_down = w_padding+1,w_padding,h_padding,h_padding
            x_padded = np.pad(X, ((0,0),(0,0),(h_padding,h_padding),(w_padding+1,w_padding)), mode="constant")
        else:
            assert h_padding+1+h_padding == h_need_pad
            assert w_padding+1+w_padding == w_need_pad
            w_pad_left,w_pad_right,h_pad_up,h_pad_down = w_padding+1,w_padding,h_padding+1,h_padding
            x_padded = np.pad(X, ((0,0),(0,0),(h_padding+1,h_padding),(w_padding+1,w_padding)), mode="constant")

    return x_padded, (w_pad_left,w_pad_right,h_pad_up,h_pad_down)



def pad_valid(X, filter_h, filter_w, stride):
    """
        new_shape = (input_shape-filter+1)/stride
        
        reference:
        https://www.jianshu.com/p/05c4f1621c7e
        """

    return X, (0,0,0,0)



def conv_forward(X, W, b, stride=1, padding="same"):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    
    if padding == "same":
        x_padded, tuple_pad = pad_same(X, h_filter, w_filter, stride)
    elif padding == "valid":
        x_padded, tuple_pad = pad_valid(X, h_filter, w_filter, stride)
    else:
        x_padded, tuple_pad = X, (0,0,0,0)
    
    

    n_x, d_x, h_x, w_x = x_padded.shape


    out_h, out_w = int((h_x - h_filter) / stride + 1), int((w_x - w_filter) / stride + 1)


    X_col = im2col_indices(x_padded, h_filter, w_filter, stride=stride)
    W_col = W.reshape(n_filters, -1)
    
    
    out = W_col @ X_col + b
    out = out.reshape(n_filters, out_h, out_w, n_x)
    out = out.transpose(3, 0, 1, 2)

        
    cache = (X, x_padded.shape, tuple_pad, W, b, stride, padding, X_col)
    
    return out, cache

"""
    1 padding and 1 stride will keep the input size
"""
def conv_backward(dout, cache):
    X, x_padded_shape, tuple_pad, W, b, stride, padding, X_col = cache
    
    w_pad_left,w_pad_right,h_pad_up,h_pad_down = tuple_pad
    
    n_filter, d_filter, h_filter, w_filter = W.shape
    
    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)
    
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ X_col.T
    dW = dW.reshape(W.shape)
    
    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped
    dX = col2im_indices(dX_col, x_padded_shape, h_filter, w_filter, stride=stride)
    
    dX = dX[:, :, h_pad_up:-h_pad_down, w_pad_left:-w_pad_right]
    
    return dX, dW, db







class Convolution_2D(Layer):
    
    def __init__(self,
                 name,
                 filter_shape: Tuple[int],
                 padding = "same",
                 stride: int = 1) -> None:
        super().__init__(name)
        
        rng = np.random.RandomState(23455)
        
        # self.input_shape = input_shape
        self.filter_shape = filter_shape
        
        
        self.padding = padding
        self.stride = stride
        
        # input_d,input_h,input_w = input_shape
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        w_bound = np.sqrt(6. / ( 32 + fan_out))
        self.params["w"] = np.asarray(
                                rng.uniform(low=-w_bound, high=w_bound, size=filter_shape),
                                dtype="float64"
                            )
                            
        
        self.params["b"] = np.zeros((filter_shape[0],1), dtype="float64")
            
            
    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        out,self.cache = conv_forward(
                                    inputs,
                                    self.params['w'],
                                    self.params['b'],
                                    stride=self.stride,
                                    padding=self.padding
                                )
                                            
        return out
            
    
    def backward(self, grad: Tensor) -> Tensor:
        dx, self.grads['w'], self.grads['b'] = conv_backward(grad, self.cache)
        return dx


                                            
