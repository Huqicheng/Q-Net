import sys
sys.path.append("../")

import numpy as np
from mnist import load
from deeplearning.utils import one_hot

from deeplearning.train import train
from deeplearning.nn import NeuralNet
from deeplearning.activation import Tanh,Softmax,Sigmoid,ReLU
from deeplearning.layers import Dense,Dropout,BatchNormalization,Flatten
from deeplearning.loss import MSE, CrossEntropy
from deeplearning.optim import Momentum_SGD,SGD,AdaGrad,RMSProp,Adam
from deeplearning.evaluation import accurarcy
from deeplearning.conv import Convolution_2D
from deeplearning.pool import Max_Pool_2D, Avg_Pool_2D


def load_data():
    mnist = {}
    mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"] = load()
    mnist["training_images"] = mnist["training_images"].reshape((60000,1,28,28))
    mnist["test_images"] = mnist["test_images"].reshape((10000,1,28,28))
    mnist["training_labels"] = one_hot(mnist["training_labels"])
    mnist["test_labels"] = one_hot(mnist["test_labels"])
    return mnist



dataset = load_data()

net = NeuralNet([
                    Convolution_2D(name="conv_1", input_shape=(1,28,28),filter_shape=(10,1,3,3),padding=1,stride=1),
                    ReLU(name="relu_1"),
                    Avg_Pool_2D(name="avg_pool_1", size=2, stride=2),
                    Convolution_2D(name="conv_2", input_shape=(10,14,14),filter_shape=(20,10,3,3),padding=1,stride=1),
                    ReLU(name="relu_2"),
                    Avg_Pool_2D(name="avg_pool_2", size=2, stride=2),
                    Flatten(name="flat_1"),
                    Dense(input_size=7*7*20, output_size=10, name="dense_1"),
                    BatchNormalization(name="bn_1",input_size=10),
                    Softmax(name="softmax_1")
                 
                 
                ])

train(net, dataset["test_images"][1000:5000], dataset["test_labels"][1000:5000], num_epochs=50,loss=CrossEntropy(),optimizer=Adam())


y_test = np.argmax(dataset["test_labels"][0:1000],axis=1)
print(accurarcy(net.predict(dataset["test_images"][0:1000]), y_test))
