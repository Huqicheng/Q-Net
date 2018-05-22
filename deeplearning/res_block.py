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



"""
    The next step:
    
    the implementation of padding is so stupid now. I'm gonna use "same" or "valid" instead.
"""

def res_block(name, n_channels, n_out_channels=None, stride=None):
    n_out_channels = n_out_channels or n_channels
    stride = stride or 1
    blockname = name
    convs = Sequential(
                name = "sequential_1",
                       
                layers = [
                    Convolution_2D(name="conv_1", filter_shape=(n_out_channels,n_channels,3,3),padding=1,stride=stride),
                    SpatialBatchNormalization(name="sbn_1",input_channel=n_out_channels),
                    Convolution_2D(name="conv_2", filter_shape=(n_out_channels,n_out_channels,3,3),padding=1,stride=1),
                    SpatialBatchNormalization(name="sbn_2",input_channel=n_out_channels)
                         ]
            )
            
    # using shortcut to learn a identity map, when using stide>1, using padding
    # to reconstruct the output to the same size of input_size
    shortcut = Identity(name=blockname+"_identity_1")
    
    if stride != 1:
        shortcut = Sequential(
                              name = "sequential_3",
                              layers = [
                                Convolution_2D(name="conv_1", filter_shape=(n_out_channels,n_channels,1,1),padding=0,stride=stride),
                                SpatialBatchNormalization(name="sbn_1",input_channel=n_out_channels)
                              ]
                    )
    

    concat = Concatenation(
                name="concat_1",
                           
                modules = [
                                convs,
                                shortcut
                          ]
             )

    res = Sequential(
            name = blockname,
                     
            layers = [
                        concat,
                        Add(name="add_1"),
                        ReLU(name="relu_1")
                     ]
          )

    return res





