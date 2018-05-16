import numpy as np

def one_hot(input):
    values = input
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]
