from deeplearning.tensor import Tensor
import numpy as np
from deeplearning.func import softmax

class Loss:
    def loss(self,predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    '''
        d loss / d y
    '''
    def grad(self,predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    def loss(self,predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted-actual)**2)
    
    def grad(self,predicted: Tensor, actual: Tensor) -> Tensor:
        return (predicted-actual)


def cross_entropy(X,y):
    """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
    """
    m = y.shape[0]
    log_likelihood = -np.log(X[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss


class CrossEntropy(Loss):
    def loss(self,predicted: Tensor, actual: Tensor) -> float:
        y = actual.argmax(axis=1)
        return cross_entropy(predicted,y)
    
    def grad(self,predicted: Tensor, actual: Tensor) -> Tensor:
        """
            gradient of cross entropy
            https://www.dropbox.com/s/rxrtz3auu845fuy/Softmax.pdf?dl=0
            https://stats.stackexchange.com/questions/277203/differentiation-of-cross-entropy
        """
        return (predicted-actual)/(predicted * (1 - predicted) + np.finfo(float).eps)
