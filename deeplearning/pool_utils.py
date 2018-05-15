import numpy as np


def maxpool(X_col):
    max_idx = np.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]
    return out, max_idx



def dmaxpool(dX_col, dout_col, pool_cache):
    dX_col[pool_cache, range(dout_col.size)] = dout_col
    return dX_col
    


def avgpool(X_col):
    out = np.mean(X_col, axis=0)
    cache = None
    return out, cache
    


def davgpool(dX_col, dout_col, pool_cache):
    dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
    return dX_col
    

