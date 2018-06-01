import numpy as np

# training dataset generation
int2binary = {}


binary_dim = 8



largest_number = pow(2,binary_dim)

binary = np.unpackbits(
            np.array([range(largest_number)],dtype=np.uint8).T,axis=1
         )

for i in range(largest_number):
    int2binary[i] = binary[i]


# prepare for training data
# to fit a function of a+b = c
batch_size = 32

X = np.zeros((batch_size,2,binary_dim))
y = np.zeros((batch_size,binary_dim))
for i in range(batch_size):
    a_int = np.random.randint(largest_number/2)
    a = int2binary[a_int] # binary encoding
    b_int = np.random.randint(largest_number/2)
    b = int2binary[b_int]
    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]

    X[i,0,:] = a
    X[i,1,:] = b
    y[i,:] = c


from deeplearning.train import train
from deeplearning.nn import Sequential
from deeplearning.activation import Tanh,Softmax,Sigmoid,ReLU
from deeplearning.layers import Dense,Dropout,Flatten
from deeplearning.rnn import RNN, LastTimeStep
from deeplearning.loss import CrossEntropy, MSE
from deeplearning.optim import SGD, Adam

net = Sequential(
            name = "net",
            layers = [
                RNN(name="rnn_1", D=8, H=8),
                Sigmoid(name="sigmoid_1"),
                LastTimeStep(name="last_1"),
                Dense(name="dense_1", input_size=8, output_size=8),
                Sigmoid(name="sigmoid_5")
            ]

      )

train(net, X, y, num_epochs=5000,loss=MSE(),optimizer=Adam())



for map_name,name,param,grad in net.get_params_grads():
    print(map_name,",",name)

def binary2int(x):
    res = 0
    for i in range(x.shape[0]):
        res *= 2
        res += x[i]
    return res


target = y


## seems well haha
for x, y in zip(X, target):
    test_matrix = np.array([x])
    predicted = binary2int(np.round(net.forward(test_matrix, training=False)[0]) )
    print(binary2int(x[0]),"+",binary2int(x[1]), "=",predicted,",expected:",binary2int(y))
