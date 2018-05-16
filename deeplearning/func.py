from deeplearning.tensor import Tensor
import numpy as np
from scipy.special import expit


def softmax(X: Tensor,
            theta: float = 1.0,
            axis: int = 1 ) -> Tensor:
    """
        reference:
        https://nolanbconaway.github.io/blog/2017/softmax-numpy
        
        Compute the softmax of each element along an axis of X.
        
        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
        first non-singleton axis.
        
        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
    """
    y = X
    # multiply y against the theta parameter
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / (ax_sum + np.finfo(float).eps)

    return p



def softmax_derivative(x: Tensor) -> Tensor:
    """
        derivative of softmax function
        
        https://www.dropbox.com/s/rxrtz3auu845fuy/Softmax.pdf?dl=0
    """
    output = softmax(x)
    return output * (1 - output)


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_derivative(x: Tensor) -> Tensor:
    return 1 - tanh(x)**2




def sigmoid(x: Tensor) -> Tensor:
    output = expit(x)
    return output

def sigmoid_derivative(x: Tensor) -> Tensor:
    output = sigmoid(x)
    return output*(1-output)




def relu(x: Tensor) -> Tensor:
    return np.maximum(x, 0.0)

def relu_derivative(x: Tensor) -> Tensor:
    output = relu(x)
    output[output>0]=1
    return output









