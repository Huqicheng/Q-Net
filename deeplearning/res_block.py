from deeplearning.nn import Sequential
from deeplearning.train import train
from deeplearning.nn import NeuralNet
from deeplearning.activation import Tanh,Softmax,Sigmoid,ReLU
from deeplearning.layers import Dense,Dropout,BatchNormalization,Flatten,Identity,Concatenation,Add
from deeplearning.loss import MSE, CrossEntropy
from deeplearning.optim import Momentum_SGD,SGD,AdaGrad,RMSProp,Adam
from deeplearning.evaluation import accurarcy
from deeplearning.conv import Convolution_2D
from deeplearning.pool import Max_Pool_2D, Avg_Pool_2D


def res_block(name, n_channels, n_out_channels=None, stride=None):
    n_out_channels = n_out_channels or n_channels
    stride = stride or 1
    
    convs = Sequential([
                        Convolution_2D(name=name+"_conv_1", filter_shape=(n_channels,n_out_channels,1,3,3),padding=1,stride=stride),
                        Convolution_2D(name=name+"_conv_2", filter_shape=(n_out_channels,n_out_channels,1,3,3),padding=1,stride=stride)
                    ])
    shortcut = Identity(name=name+"_identity_1")

    concat = Concatenation(name=name+"_concat_1",
                           modules = [
                                convs,
                                shortcut
                           ])

    res = Sequential([
                        concat,
                        ReLU(name=name+"_relu_1")
                     
                     ])

    return res





