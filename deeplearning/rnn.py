import numpy as np

from deeplearning.tensor import Tensor
from deeplearning.layers import Layer
from deeplearning.func import tanh,sigmoid,sigmoid_derivative

"""
    https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
    http://manutdzou.github.io/2016/07/11/RNN-backpropagation.html
    
    https://github.com/mklissa/CS231n/tree/master/assignment3/cs231n
"""


class LastTimeStep(Layer):
    def __init__(self, name):
        """
            Forwarding the last time_step of the output to the next layer.
        """
        super().__init__(name)
    
    def forward(self, inputs, **kwargs):
        self.input_shape = inputs.shape
        return inputs[:,-1,:]
    
    def backward(self, grad):
        dx = np.zeros(self.input_shape)
        dx[:,-1,:] = grad
        return dx


class RNN_Cell(Layer):
    def __init__(self, name, D, H):
        super().__init__(name)
        
        self.D = D
        self.H = H

        self.params["Wxh"] = np.random.randn(D, H) / np.sqrt(D / 2.)
        self.params["Whh"] = np.random.randn(H, H) / np.sqrt(H / 2.)
        self.params["bh"] = np.zeros((1,H))
    
    

    def forward(self, inputs, **kwargs):
        state = kwargs["cache"]
        
        Wxh, Whh = self.params['Wxh'], self.params['Whh']
        bh = self.params['bh']

        h_previous = state.copy()

        gate_input = inputs @ Wxh + h_previous @ Whh + bh
        cache = (gate_input, inputs, h_previous)
        
        return gate_input, cache


    def backward(self, grad, **kwargs):
        Wxh, Whh = self.params['Wxh'], self.params['Whh']
        bh = self.params['bh']
        gate_input, inputs, h_previous = kwargs["cache"]
        
        dinputs =  grad @ Wxh.T
        
        self.grads["Wxh"] = inputs.T @ grad
        self.grads["Whh"] = h_previous.T @ grad
        self.grads["bh"] = np.sum(grad,axis=0)
    
        dprev_h = grad @ Whh.T
        
        return dinputs, dprev_h



class RNN(Layer):
    def __init__(self, name, D, H):
        """
            Contains one rnn cell and get output by "forwarding" the cell recurrently.
            
            Params:
            ---------------
            D: input size (size of the last dimension of the input matrix)
            
            H: size of hidden layer
        """
        super().__init__(name)
        
        self.D = D
        self.H = H

        self.rnn_cell = RNN_Cell(name,D,H)


    def forward(self, inputs, **kwargs):
        N,T,H = inputs.shape
        self.N = N
        self.T = T
        
        # initiate some caches
        h = np.zeros((N,T,H))
        h0 = np.zeros((N,H))
        self.time_cache = [None for i in range(T)]

        prev_h = h0
        
        # iterate from the first cell to the last one (actually, the same one, because they share weights)
        for t in range(T):
            next_h,self.time_cache[t] = self.rnn_cell.forward(inputs[:,t,:].squeeze(),cache=(prev_h))
            h[:,t,:] = next_h
            prev_h = next_h
        
        # the output seems like (batch_size, time_steps, size_of_hidden_layer)
        # for some classification problems, just using the last time_step, aka, h[:,-1,:]
        return h

    def backward(self, grad):
        D,T,H = self.D, self.T, self.H
        
        # initiation
        dWxh = np.zeros((D, H))
        dWhh = np.zeros((H, H))
        dbh = np.zeros((H))
        dx = np.zeros((self.N, T, D))
        dprev_h_t = np.zeros((self.N,H))
        
        # back-prop with respect to the time step
        for t in reversed(range(T)):
            cache = (self.time_cache[t][0], self.time_cache[t][1], self.time_cache[t][2])
            dx[:,t,:], dprev_h_t = self.rnn_cell.backward(grad[:,t,:] + dprev_h_t, cache=cache)
            dWxh += self.rnn_cell.grads["Wxh"]
            dWhh += self.rnn_cell.grads["Whh"]
            dbh += self.rnn_cell.grads["bh"]

        dh0 = dprev_h_t

        self.rnn_cell.grads["Wxh"] = dWxh
        self.rnn_cell.grads["Whh"] = dWhh
        self.rnn_cell.grads["bh"] = dbh

        return dx


    def get_params_grads(self):
        """
            return  (name in the map of optimizer, real param name, param)
        """
        for map_name,name, param, grad in self.rnn_cell.get_params_grads():
            yield self.name+'_'+map_name, name, param, grad





















