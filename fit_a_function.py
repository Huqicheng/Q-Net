import numpy as np


from deeplearning.nn import NeuralNet
from deeplearning.activation import Tanh,Sigmoid
from deeplearning.layers import Dense
from deeplearning.train import train

inputs = np.array([
                   [0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1],
                   [2,2]
                   ])

targets = np.array([
                    [0],
                    [1],
                    [1],
                    [2],
                    [6]
                    ])

net = NeuralNet([
                 Dense(input_size=2, output_size=50),
                 Sigmoid(),
                 Dense(input_size=50, output_size=1)
                 ])

train(net, inputs, targets, num_epochs=10000)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
