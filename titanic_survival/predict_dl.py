import sys

sys.path.append('../')



import numpy as np
from dataset import *
from sklearn import neighbors
from sklearn import svm
from sklearn.gaussian_process.kernels import RBF

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

ds = dataset()
x,y,x_train, x_test, y_train, y_test = ds.get_data()

print (x_train.shape)
print (y_train.shape)

import numpy as np

from deeplearning.train import train
from deeplearning.nn import NeuralNet
from deeplearning.activation import Tanh,Softmax,Sigmoid,ReLU
from deeplearning.layers import Dense,Dropout,BatchNormalization
from deeplearning.loss import MSE, CrossEntropy
from deeplearning.optim import Momentum_SGD,SGD,AdaGrad,RMSProp,Adam
from deeplearning.evaluation import accurarcy


net = NeuralNet([
    Dense(input_size=12, output_size=50,name="dense_1"),
    BatchNormalization(input_size=50,name="bn_1"),
    ReLU(name="relu_1"),
    Dense(input_size=50, output_size=100,name="dense_2"),
    BatchNormalization(input_size=100,name="bn_2"),
    ReLU(name="relu_2"),
    Dense(input_size=100, output_size=2,name="dense_4"),
    BatchNormalization(input_size=2,name="bn_4"),
    Softmax(name="softmax_1")
])

train(net, x_train, y_train, num_epochs=1000,loss=CrossEntropy(),optimizer=Adam())

y_test = np.argmax(y_test,axis=1)
print(accurarcy(net.predict(x_test), y_test))





