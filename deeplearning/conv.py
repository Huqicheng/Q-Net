import numpy as np
from deeplearning.layers import Layer
from deeplearning.conv_utils import *
from deeplearning.tensor import Tensor
from typing import Tuple

"""
    Stanford CS231n
    
    https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
    
    http://cs231n.github.io/convolutional-networks/
"""




def pad_same(X, filter_h, filter_w, stride):
    """
        new_shape = input_shape/stride
        
        params:
        
        X: input matrix
            (batch_size,channel_num,height,width)
        
        filter_h: height of the filter
        
        filter_w: width of the filter
        
        stride: stride of convolution
        
        return:
        
        x_padded: the input matrix after padding
        
        tuple_padding: the padding number for height and width 
                    (w_pad_left,w_pad_right,h_pad_up,h_pad_down)
        
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

    w_pad_left,w_pad_right,h_pad_up,h_pad_down = w_padding,w_padding,h_padding,h_padding
    
    w_pad_left = w_pad_left + 1 if w_need_pad % 2 != 0 else w_pad_left
    h_pad_up = h_pad_up + 1 if h_need_pad % 2 != 0 else h_pad_up

    x_padded = np.pad(X, ((0,0),(0,0),(h_pad_up,h_pad_down),(w_pad_left,w_pad_right)), mode="constant")
    
    return x_padded, (w_pad_left,w_pad_right,h_pad_up,h_pad_down)



def pad_valid(X, filter_h, filter_w, stride):
    """
        new_shape = (input_shape-filter+1)/stride
        
        return:
        
        x_padded: the input matrix after padding
        
        tuple_padding: the padding number for height and width
                    (w_pad_left,w_pad_right,h_pad_up,h_pad_down)
        
        reference:
        https://www.jianshu.com/p/05c4f1621c7e
    """

    return X, (0,0,0,0)



def conv_forward(X, W, b, stride=1, padding="same"):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    
    # padding
    if padding == "same":
        x_padded, tuple_pad = pad_same(X, h_filter, w_filter, stride)
    elif padding == "valid":
        x_padded, tuple_pad = pad_valid(X, h_filter, w_filter, stride)
    else:
        x_padded, tuple_pad = X, (0,0,0,0)

    # new shape after padding
    n_x, d_x, h_x, w_x = x_padded.shape

    # expected output shape
    out_h, out_w = int((h_x - h_filter) / stride + 1), int((w_x - w_filter) / stride + 1)

    # check if the output shape is legal
    if h_filter != 1:
        assert (h_x - h_filter) % stride == 0
        assert (w_x - w_filter) % stride == 0
    else:
        assert h_x % stride == 0
        assert w_x % stride == 0

    # do convolution
    X_col = im2col_indices(x_padded, h_filter, w_filter, stride=stride)
    W_col = W.reshape(n_filters, -1)
    out = W_col @ X_col + b
    out = out.reshape(n_filters, out_h, out_w, n_x)
    out = out.transpose(3, 0, 1, 2)

    # push to cache
    cache = (X, x_padded.shape, tuple_pad, W, b, stride, padding, X_col)
    
    return out, cache


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
        """
            
            params:
            
            filter_shape: tuple of intger
                    (channel_num, filter_height, filter_width)
                    
            padding: padding mode default as "same"
                    "same" - output_size = input_size / stride
                    "valid" - output_size = (input_shape - filter_size + 1) / stride
                    
            stride: stride of the convolution
                    default as 1
                
        """
        super().__init__(name)
        
        rng = np.random.RandomState(23455)
        
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride

        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        w_bound = np.sqrt(6. / ( 32 + fan_out))
        
        # notice: when using gpu, only float32 is allowed
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


                                            
