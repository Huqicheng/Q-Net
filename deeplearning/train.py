from deeplearning.tensor import Tensor
from deeplearning.nn import NeuralNet,Sequential
from deeplearning.loss import Loss, MSE
from deeplearning.optim import Optimizer, SGD
from deeplearning.data import DataIterator, BatchIterator
from deeplearning.evaluation import accurarcy


def train(net: Sequential,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 200,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD(lr=0.2)) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs, training=True)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
