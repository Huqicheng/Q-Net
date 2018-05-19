from deeplearning.nn import Sequential
from deeplearning.train import train
from deeplearning.nn import NeuralNet
from deeplearning.activation import Tanh,Softmax,Sigmoid,ReLU
from deeplearning.layers import Dense,Dropout,BatchNormalization,SpatialBatchNormalization,Flatten,Identity,Concatenation,Add
from deeplearning.loss import MSE, CrossEntropy
from deeplearning.optim import Momentum_SGD,SGD,AdaGrad,RMSProp,Adam
from deeplearning.evaluation import accurarcy
from deeplearning.conv import Convolution_2D
from deeplearning.pool import Max_Pool_2D, Avg_Pool_2D


def res_block(name, n_channels, n_out_channels=None, stride=None):
    n_out_channels = n_out_channels or n_channels
    stride = stride or 1
    blockname = name
    convs = Sequential(
                       name = blockname+"_sequential_1",
                       layers = [
                        Convolution_2D(name=blockname+"_conv_1", filter_shape=(n_out_channels,n_channels,3,3),padding=1,stride=stride),
                        SpatialBatchNormalization(name=blockname+"_sbn_1",input_channel=n_out_channels),
                        Convolution_2D(name=blockname+"_conv_2", filter_shape=(n_out_channels,n_out_channels,3,3),padding=1,stride=stride),
                        SpatialBatchNormalization(name=blockname+"_sbn_2",input_channel=n_out_channels)
                        ]
                       )
    shortcut = Identity(name=blockname+"_identity_1")

    concat = Concatenation(name=blockname+"_concat_1",
                           modules = [
                                convs,
                                shortcut
                           ])

    res = Sequential(
                     name = blockname+"_sequential_2",
                     layers = [
                        concat,
                        Add(name=blockname+"_add_1"),
                        ReLU(name=blockname+"_relu_1")
                     
                     ]
                )

    return res





