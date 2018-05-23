import numpy as np


def maxpool(X_col):
    max_idx = np.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]
    return out, max_idx


"""
    Explanation:
    
        Only the maximum within a window is kept as the input to the next layer,
        so only the maximum will influence the next layer in order to influence the whole NN,
        so just back-prop the gradient of the maximum entry and ignore others.
"""
def dmaxpool(dX_col, dout_col, pool_cache):
    dX_col[pool_cache, range(dout_col.size)] = dout_col
    return dX_col
    


def avgpool(X_col):
    out = np.mean(X_col, axis=0)
    cache = None
    return out, cache
    

"""
    Explanation:
    
        All of entries in the window are used to calculate the mean, 
        so all of them will influence the next layer,
        so back-prop the dout/size as the gradient on each entry, where size is size of the window
"""
def davgpool(dX_col, dout_col, pool_cache):
    dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
    return dX_col
    

