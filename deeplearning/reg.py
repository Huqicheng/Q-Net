import numpy as np

class Regularization:
    def __init__(self,reg_func,lamda):
        self.lamda = lamda
        self.reg_func = reg_func

    def regularize(self,W):
        return self.reg_func(W,self.lamda)


def l2_reg(W, lam=1e-3):
    return .5 * lam * np.sum(W * W)


def dl2_reg(W, lam=1e-3):
    return lam * W


def l1_reg(W, lam=1e-3):
    return lam * np.sum(np.abs(W))


def dl1_reg(W, lam=1e-3):
    return lam * W / (np.abs(W) + np.finfo(float).eps)


class L2_Regularization(Regularization):
    def __init__(self,lamda=1e-3):
        super().__init__(dl2_reg,lamda)


class L1_Regularization(Regularization):
    def __init__(self,lamda=1e-3):
        super().__init__(dl1_reg,lamda)


