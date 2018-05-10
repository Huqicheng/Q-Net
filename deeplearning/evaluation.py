import numpy as np
from sklearn.metrics import mean_squared_error

def MSE(predicted,target):
    return mean_squared_error(predicted, target)

def RMSE(predicted,target):
    return np.sqrt(mean_squared_error(predicted, target))

def errorRatio(predicted,target):
    return np.mean(np.not_equal(predicted,target))

def accurarcy(predicted,target):
    return np.mean(np.equal(predicted,target))



