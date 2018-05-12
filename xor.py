import numpy as np

from deeplearning.train import train
from deeplearning.nn import NeuralNet
from deeplearning.activation import Tanh,Softmax,Sigmoid,ReLU
from deeplearning.layers import Dense,Dropout,BatchNormalization
from deeplearning.loss import CrossEntropy
from deeplearning.optim import Momentum_SGD,SGD,AdaGrad,RMSProp,Adam

inputs = np.array([
                   [0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1],
                   [0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1],
                   [0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1]
                   ], dtype='float64')

targets = np.array([
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 0]
                    ], dtype='float64')

net = NeuralNet([
    Dense(input_size=2, output_size=1000, name="dense_1"),
    BatchNormalization(name="bn_1",input_size=1000),
    Sigmoid(name="sigmoid_1"),
    Dense(input_size=1000, output_size=2,name="dense_2"),
    BatchNormalization(name="bn_2",input_size=2),
    Softmax(name="softmax_1")
])



train(net, inputs, targets, num_epochs=200,loss=CrossEntropy(),optimizer=Adam())

# train(net, inputs, targets, num_epochs=3000)

for x, y in zip(inputs, targets):
    test_matrix = np.array([x])
    predicted = net.predict(test_matrix)
    predicted_prob = net.predict_prob(test_matrix)
    print(x, predicted, predicted_prob,y)


