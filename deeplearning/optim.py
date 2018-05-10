from deeplearning.nn import NeuralNet

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
