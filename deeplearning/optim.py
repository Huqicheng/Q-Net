from deeplearning.nn import NeuralNet
import numpy as np

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

"""
    references:
    https://blog.csdn.net/u010089444/article/details/76725843
    https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
"""

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, net: NeuralNet) -> None:
        """
            vi = alpha * grad
            w = w - vi
        """
        for name, param, grad in net.params_and_grads():
            param -= self.lr * grad


class Momentum_SGD(Optimizer):
    def __init__(self, lr: float = 0.01, gamma: float = 0.9) -> None:
        self.lr = lr
        self.gamma = gamma
        self.velocity = {}
    
    def get_velocity(self, name):
        if name in self.velocity:
            return self.velocity[name]
        else:
            return 0

    def step(self, net: NeuralNet) -> None:
        """
            vi = gamma * vi-1 + alpha * grad
            w = w - vi
        """
        for name, param, grad in net.params_and_grads():
            self.velocity[name] = self.gamma * self.get_velocity(name) + self.lr * grad
            param -= self.velocity[name]




class AdaGrad(Optimizer):
    """
        https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
    
        The problem with learning rate in Gradient Descent is that it’s constant and affecting all of our parameters.
        What happen if we know that we should slow down or speed up?
        What happen if we know that we should accelerate more in this direction and decelerate in that direction?
    """
    def __init__(self, lr: float = 0.02) -> None:
        self.lr = lr
        self.cache = {}
        self.eps = np.finfo(float).eps

    def get_cache(self, name):
        if name in self.cache:
            return self.cache[name]
        else:
            return 0

    def step(self, net: NeuralNet) -> None:
        """
        
        """
        for name, param, grad in net.params_and_grads():
            self.cache[name] = self.get_cache(name) + grad**2
            param -= self.lr * grad / (np.sqrt(self.cache[name]) + self.eps)






class RMSProp(Optimizer):
    """
        https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
        
        AdaGrad could be problematic as the learning rate will be monotonically decreasing to the point that the learning stops altogether because of the very tiny learning rate.
        
        RMSprop decay the past accumulated gradient, so only a portion of past gradients are considered. Now, instead of considering all of the past gradients, RMSprop behaves like moving average.
    """
    def __init__(self, lr: float = 0.02, gamma: float = 0.9) -> None:
        self.lr = lr
        self.cache = {}
        self.gamma = gamma
        self.eps = np.finfo(float).eps

    def get_cache(self, name):
        if name in self.cache:
            return self.cache[name]
        else:
            return 0

    def step(self, net: NeuralNet) -> None:
        
        for name, param, grad in net.params_and_grads():
            self.cache[name] = self.gamma * self.get_cache(name) + (1 - self.gamma) * (grad**2)
            param -= self.lr * grad / (np.sqrt(self.cache[name]) + self.eps)



class Adam(Optimizer):
    """
        https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
        
        Adam is the latest state of the art of first order optimization method that’s widely used in the real world. It’s a modification of RMSprop. Loosely speaking, Adam is RMSprop with momentum. So, Adam tries to combine the best of both world of momentum and adaptive learning rate.
    """
    def __init__(self,
                 lr: float = 0.02,
                 gamma_v: float = 0.9,
                 gamma_c: float = 0.9) -> None:
        self.lr = lr
        self.cache = {}
        self.velocity = {}
        self.step_num = {}
        self.gamma_v = gamma_v
        self.gamma_c = gamma_c
        self.eps = np.finfo(float).eps

    def get_cache(self, name):
        if name in self.cache:
            return self.cache[name]
        else:
            return 0

    def get_velocity(self, name):
        if name in self.velocity:
            return self.velocity[name]
        else:
            return 0

    def get_step(self, name):
        if name in self.step_num:
            return self.step_num[name]
        else:
            return 0

    def step(self, net: NeuralNet) -> None:
    
        for name, param, grad in net.params_and_grads():
            self.velocity[name] = self.get_velocity(name) * self.gamma_v + (1 - self.gamma_v) * grad
            self.cache[name] = self.get_cache(name) * self.gamma_c + (1 - self.gamma_c) * grad**2
            
            self.step_num[name] = self.get_step(name) + 1
            
            v_hat = self.velocity[name] / (1.0 - self.gamma_v**(self.step_num[name]))
            c_hat = self.cache[name] / (1.0 - self.gamma_c**(self.step_num[name]))
            
            param -= self.lr * v_hat / (np.sqrt(c_hat) + self.eps)







