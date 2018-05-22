import sys
sys.path.append("../")

import numpy as np
from mnist import load
from deeplearning.utils import one_hot

from deeplearning.train import train
from deeplearning.nn import NeuralNet,Sequential
from deeplearning.activation import Tanh,Softmax,Sigmoid,ReLU
from deeplearning.layers import Dense,Dropout,BatchNormalization,Flatten,SpatialBatchNormalization
from deeplearning.loss import MSE, CrossEntropy
from deeplearning.optim import Momentum_SGD,SGD,AdaGrad,RMSProp,Adam
from deeplearning.evaluation import accurarcy
from deeplearning.conv import Convolution_2D
from deeplearning.pool import Max_Pool_2D, Avg_Pool_2D
from deeplearning.res_block import res_block


def load_data():
    mnist = {}
    mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"] = load()
    mnist["training_images"] = mnist["training_images"].reshape((60000,1,28,28))
    mnist["test_images"] = mnist["test_images"].reshape((10000,1,28,28))
    mnist["training_labels"] = one_hot(mnist["training_labels"])
    mnist["test_labels"] = one_hot(mnist["test_labels"])
    return mnist



dataset = load_data()

net = Sequential(
                 name = "residual_net",
                 layers = [
                    res_block(name="res_block_1",n_channels=1,n_out_channels=5,stride=2),
                    Flatten(name="flat_1"),
                    Dense(input_size=14*14*5, output_size=10, name="dense_1"),
                    BatchNormalization(name="bn_1",input_size=10),
                    Softmax(name="softmax_1")
                 
                 
                ])

train(net, dataset["test_images"][1000:5000], dataset["test_labels"][1000:5000], num_epochs=20,loss=CrossEntropy(),optimizer=Adam())


y_test = np.argmax(dataset["test_labels"][0:1000],axis=1)
print(accurarcy(net.predict(dataset["test_images"][0:1000]), y_test))

for map_name,name,param,grad in net.get_params_grads():
    print(map_name,",",name)
