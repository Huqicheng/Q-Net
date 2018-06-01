import sys
sys.path.append("../")


from deeplearning.tensor import Tensor
from deeplearning.nn import NeuralNet,Sequential
from deeplearning.loss import Loss, MSE
from deeplearning.optim import Optimizer, SGD
from img_utils import CaptchaBatchIterator, next_batch
from deeplearning.evaluation import accurarcy

import numpy as np
from deeplearning.train import train
from deeplearning.nn import NeuralNet
from deeplearning.activation import Tanh,Softmax,Sigmoid,ReLU
from deeplearning.layers import Dense,Dropout,BatchNormalization,Flatten,SpatialBatchNormalization
from deeplearning.loss import MSE, CrossEntropy
from deeplearning.optim import Momentum_SGD,SGD,AdaGrad,RMSProp,Adam
from deeplearning.evaluation import accurarcy
from deeplearning.conv import Convolution_2D
from deeplearning.pool import Max_Pool_2D, Avg_Pool_2D

from gen_data import *


batch_validate, batch_validate = next_batch(200)
batch_validate = batch_inputs.reshape((200,1,CAPTCHA_HEIGHT,CAPTCHA_WIDTH))

def train(net: NeuralNet,
          num_epochs: int = 200,
          iterator: CaptchaBatchIterator = CaptchaBatchIterator(32),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD(lr=0.2)) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(30):
            predicted = net.forward(batch.inputs, training=True)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        
        print(epoch, epoch_loss)


net = NeuralNet([
                    Convolution_2D(name="conv_1", filter_shape=(10,1,3,3),padding="same",stride=1),
                    Avg_Pool_2D(name="avg_pool_1", size=2, stride=2),
                    SpatialBatchNormalization(name="sbn_1",input_channel=10),
                    ReLU(name="relu_1"),
                    Convolution_2D(name="conv_2", filter_shape=(20,10,3,3),padding="same",stride=1),
                    Avg_Pool_2D(name="avg_pool_2", size=2, stride=2),
                    SpatialBatchNormalization(name="sbn_2",input_channel=20),
                    ReLU(name="relu_2"),
                    Flatten(name="flat_1"),
                    Dense(input_size=15*40*20, output_size=100, name="dense_1"),
                    BatchNormalization(name="bn_1",input_size=100),
                    ReLU(name="relu_3"),
                    Dense(input_size=100, output_size=40, name="dense_2"),
                    BatchNormalization(name="bn_2",input_size=40),
                    Sigmoid(name="sigmoid_1")


                ])

train(net, num_epochs = 500)

