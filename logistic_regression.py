import numpy as np

from deeplearning.train import train
from deeplearning.nn import NeuralNet
from deeplearning.activation import Tanh,Softmax,Sigmoid,ReLU
from deeplearning.layers import Dense,Dropout
from deeplearning.loss import CrossEntropy
from deeplearning.optim import Momentum_SGD

inputs = np.array([
                   [0, 0],
                   [0, 1],
                   [1, 1],
                   [1, 0]
                   ])

targets = np.array([
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 0]
                    ])

net = NeuralNet([
                 Dense(input_size=2, output_size=2, name="dense_1"),
                 Softmax(name="softmax_1"),
                 ])



train(net, inputs, targets, num_epochs=1000,loss=CrossEntropy(),optimizer=Momentum_SGD())

# train(net, inputs, targets, num_epochs=3000)

for x, y in zip(inputs, targets):
    predicted = net.forward(x,training=False)
    print(x, predicted, y)
