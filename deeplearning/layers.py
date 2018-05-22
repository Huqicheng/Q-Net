from deeplearning.tensor import Tensor
from typing import Sequence, Tuple

"""
    tricks of DL
    
    http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
"""



import numpy as np
from typing import Dict,Tuple

class Layer:
    """
        The abstract class of all kinds of layers.
    """

    def __init__(self, name) -> None:
        self.params: Dict[Str,Tensor] = {}
        self.grads: Dict[Str,Tensor] = {}
        self.name = name

    def forward(self,inputs: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def backward(self,grad: Tensor) -> Tensor:
        raise NotImplementedError

    def get_params_grads(self):
        for name, param in self.params.items():
            yield name, name ,param, self.grads[name]

class Identity(Layer):
    """
        output = input
    """
    def __init__(self,name) -> None:
        super().__init__(name)

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return inputs
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad

class Dense(Layer):

    '''
    The fully-connected layer.
    
    output = inputs@w+b
        
    inputs = (batch_size,input_size)
    outputs = (batch_size,output_size)
    '''
    def __init__(self,
                 name,
                 input_size: int,
                 output_size: int) -> None:
        super().__init__(name)
        self.params["w"] = np.float64(np.random.randn(input_size,output_size))
        self.params["b"] = np.float64(np.random.randn(output_size))


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
        Randomly ignore some signals while training.
        
        reference:
        1. https://blog.csdn.net/stdcoutzyx/article/details/49022443
        2. https://blog.csdn.net/hjimce/article/details/50413257
    '''
    def __init__(self,
                 name,
                 dropout_rate=0.5) -> None:
        super().__init__(name)
        self.dropout_rate = dropout_rate

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        is_training = kwargs['training']
        if is_training:
            self.mask = np.float64(np.random.binomial(n=1,p=1-self.dropout_rate,size=inputs.shape[1]))
        else:
            self.mask = 1.
        return inputs*self.mask/(1-self.dropout_rate)

    def backward(self, grad: Tensor) -> Tensor:
        return grad*self.mask/(1-self.dropout_rate)

class Flatten(Layer):
    def __init__(self,
                 name) -> None:
        super().__init__(name)
        

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        self.input_shape = inputs.shape
        return inputs.flatten().reshape(inputs.shape[0],-1)

    def backward(self, grad: Tensor) -> Tensor:
        return grad.reshape(self.input_shape)


class Concatenation(Layer):
    """
        Gather the output of each module as a list.
    """
    def __init__(self,name,
                 modules: Sequence[Layer]) -> None:
        super().__init__(name)
        self.modules = modules

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        output = []
        for module in self.modules:
            output.append(module.forward(inputs,training=kwargs["training"]))

        return output

    def backward(self, grad: Tensor) -> Tensor:
        dx = 0
        for idx,g in enumerate(grad):
            self.modules[idx].backward(g)
            if idx == 0:
                dx = g
            else:
                dx += g
        return dx

    def get_params_grads(self):
        """
            return  (name in the map of optimizer, real param name, param)
        """
        for module in self.modules:
            for map_name,name, param, grad in module.get_params_grads():
                yield module.name+'_'+map_name, name, param, grad


class Add(Layer):
    """
        Add the output of each modules, always used after Concatenation.
    """
    def __init__(self,name) -> None:
        super().__init__(name)
            
    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        output = 0
        self.module_num = len(inputs)
        for idx,input in enumerate(inputs):
            if idx == 0:
                output = input
            else:
                output += input
        return output

    def backward(self, grad: Tensor) -> Tensor:
        dx = []
        for i in range(self.module_num):
            dx.append(np.copy(grad))
        return dx


class Padding(Layer):
    """
        Padding the input to the indicated shape.
    """
    def __init__(self,name,
                 dim: Tuple[int],
                 pad: int) -> None:
        super().__init__(name)
        self.dim = dim
        self.pad = pad
    
    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        pads = []
        self.ndim = inputs.ndim
        for axis in range(inputs.ndim):
            if axis in self.dim:
                pads += [(self.pad, self.pad)]
            else:
                pads += [(0,0)]
        pads = tuple(pads)
        output = np.pad(inputs, pads, mode="constant")
        return output
    
    def backward(self, grad: Tensor) -> Tensor:
        slc = [slice(None)] * self.ndim
        dx = grad
        for axis in range(self.ndim):
            if axis in self.dim:
                slc[axis] = slice(self.pad, -self.pad)
        dx = grad[slc]
        return dx



class BatchNormalization(Layer):
    """
        Batch Normalization Layer for accelerating training of the neural network and stablizing the training process.
        
        Training:
            normedX = input-mean/sqrt(var+eps)
            output = gamma*normedX+beta
            global_mean = (1-decay)*mean+decay*global_mean
            global_var = (1-decay)*var+decay*global_var
            
        Testing:
            normedX = input-global_mean/sqrt(global_var+eps)
            output = gamma*normedX+beta
        
        reference:
        1. https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        2. https://www.quora.com/How-does-batch-normalization-behave-differently-at-training-time-and-test-time
        3. http://dengyujun.com/2017/09/30/understanding-batch-norm/
        4. https://arxiv.org/pdf/1502.03167v3.pdf
    """
    def __init__(self,
                 name,
                 input_size: int,
                 decay: float = 0.9) -> None:
        super().__init__(name)
        self.params['gamma'] = np.ones((input_size,), dtype='float64')
        self.params['beta'] = np.zeros((input_size,), dtype='float64')
        self.global_mean = np.zeros((input_size,), dtype='float64')
        self.global_var = np.zeros((input_size,), dtype='float64')
        self.decay = decay
        self.input_size = input_size
        self.eps = np.finfo(float).eps

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        training = kwargs["training"]
        
        # while testing, using global mean and global variance
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
        
        # Estimation of glocal mean and variance
        self.global_mean = self.global_mean * self.decay + mu * (1 - self.decay)
        self.global_var = self.global_var * self.decay + self.var * (1 - self.decay)
        
        return self.out


#    def backward(self, dy: Tensor) -> Tensor:
#        N,D = dy.shape
#        h = self.out
#        mu = 1./N*np.sum(h, axis = 0)
#        var = 1./N*np.sum((h-mu)**2, axis = 0)
#        self.grads['beta'] = np.sum(dy, axis=0)
#        self.grads['gamma'] = np.sum((h - mu) * (var + np.finfo(float).eps)**(-1. / 2.) * dy, axis=0)
#        dh = (1. / N) * self.params['gamma'] * (var + np.finfo(float).eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0) - (h - mu) * (var + np.finfo(float).eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))
#        return dh


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


class SpatialBatchNormalization(BatchNormalization):
    """
        Batch Normalization for Convolution Layers.
        
        N*H*W as input of the usual BN because the convolutional layer has a special property: filter weights are shared across the input image so let the output of the same filter to be transformed by the the same way(bn weights).
        
        For convolutional layers, we additionally want the normalization to obey the convolutional property – so that different elements of the same feature map, at different locations, are normalized in the same way. To achieve this, we jointly normalize all the activations in a mini- batch, over all locations. In Alg. 1, we let B be the set of all values in a feature map across both the elements of a mini-batch and spatial locations – so for a mini-batch of size m and feature maps of size p × q, we use the effec- tive mini-batch of size m′ = |B| = m · pq. We learn a pair of parameters γ(k) and β(k) per feature map, rather than per activation. Alg. 2 is modified similarly, so that during inference the BN transform applies the same linear transformation to each activation in a given feature map.
        
        reference:
        1. https://stackoverflow.com/questions/38553927/batch-normalization-in-convolutional-neural-network
        2. https://arxiv.org/pdf/1502.03167v3.pdf
        3. http://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays
    """
    def __init__(self,
                 name,
                 input_channel: int,
                 decay: float = 0.99) -> None:
        """
            params:
                input_channel: number of channels of input data
                decay: momentum factor for running_mean and running variance calculation.
        """

        super().__init__(name,input_channel,decay)

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        N,C,H,W = inputs.shape
        x_flat = inputs.transpose(0, 2, 3, 1).reshape(-1, C)
        x_flat = np.ascontiguousarray(x_flat,dtype=inputs.dtype)
        output = super().forward(x_flat,training=kwargs["training"])
        output = output.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return output

    def backward(self, grad: Tensor) -> Tensor:
        N,C,H,W = grad.shape
        dout_flat = grad.transpose(0, 2, 3, 1).reshape(-1, C)
        dout_flat = np.ascontiguousarray(dout_flat, dtype=dout_flat.dtype)
        dx = super().backward(dout_flat)
        dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return dx









