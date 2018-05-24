import numpy as np


def l2_reg(W, lam=1e-3):
    return .5 * lam * np.sum(W * W)


def dl2_reg(W, lam=1e-3):
    return lam * W


def l1_reg(W, lam=1e-3):
    return lam * np.sum(np.abs(W))


def dl1_reg(W, lam=1e-3):
    return lam * W / (np.abs(W) + np.finfo(float).eps)
