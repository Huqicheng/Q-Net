from deeplearning.tensor import Tensor
import numpy as np


def softmax(x: Tensor) -> Tensor:
    exps = np.exp(x.T)
    sum = exps.sum(axis=0)
    res = exps / sum
    res = res.T
    return res

def softmax_grad(o: Tensor) -> Tensor:
    s = o.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

#def softmax_derivative(x: Tensor) -> Tensor:
#    output = softmax(x)
#    assert output.shape == x.shape
#    gradient = np.zeros((x.shape[0],output.shape[1],x.shape[1]))
#    for s in range(output.shape[0]):
#        gradient[s] = softmax_grad(output[s])
#    return gradient.mean(axis=1)



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



'''
    it's right!
'''
def sigmoid(x: Tensor) -> Tensor:
    output = 1/(1+np.exp(-x))
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
