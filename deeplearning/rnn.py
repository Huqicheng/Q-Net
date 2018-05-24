import numpy as np

from deeplearning.tensor import Tensor
from deeplearning.layers import Layer
from deeplearning.func import tanh,softmax

"""
    https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
    http://manutdzou.github.io/2016/07/11/RNN-backpropagation.html
"""
class RNN_Cell(Layer):
    def __init__(self, name, D, H):
        super.__init__(name)
        
        self.D = D
        self.H = H

        self.params["Wxh"] = np.random.randn(D, H) / np.sqrt(D / 2.)
        self.params["Whh"] = np.random.randn(H, H) / np.sqrt(H / 2.)
        self.params["Why"] = np.random.randn(H, D) / np.sqrt(D / 2.)

        self.params["bh"] = np.zeros((1,H))
        self.params["by"] = np.zeros((1,D))

    def forward(self, inputs, **kwargs):
        training = kwargs["training"]
        
        Wxh, Whh, Why = self.params['Wxh'], self.params['Whh'], self.params['Why']
        bh, by = self.params['bh'], self.params['by']

        previous = inputs["pre"].copy()
        input = inputs["input"].copy()
        
        X_one_hot = np.zeros(self.D)
        X_one_hot[input] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)

        h = tanh(X_one_hot @ Wxh + previous @ whh + bh)
        y = h @ Why + by
        
        if not training:
            y = softmax(y)

        self.cache = (X_one_hot, h, previous, y)
        return y


    def backward(self, grad):
        Wxh, Whh, Why = self.params['Wxh'], self.params['Whh'], self.params['Why']
        bh, by = self.params['bh'], self.params['by']




