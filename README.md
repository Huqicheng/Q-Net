DeepLearningFramework
=====
A simplified deep learning framework.<br>

## Requirements
* Python 3.6
* Numpy 0.14.0
* Scipy 1.0.0


## 1. Layers
* Dense
* Dropout
* Activations
  * Sigmoid
  * Tanh
  * Softmax
  * ReLU
* BatchNormalization
  * SpatialBatchNormalization
* Pool_2D
  * Avg_Pool
  * Max_Pool
* Convolution_2D
* Concatenation + Add
* Residual Block

  

## 2. Loss Functions
* MSE
* CrossEntropy

## 3. Optimizers
* SGD 
* Momemtum_SGD 
* AdaGrad
* RMSProp
* Adam

## 4. Examples

#### 4.1 xor
The source code of training a Neural Network to fit the xor function is in xor.py.
#### 4.1.1 Building a Neural Network with one hidden layer and one softmax layer.
```
net = NeuralNet([
    Dense(input_size=2, output_size=20, name="dense_1"),
    BatchNormalization(name="bn_1",input_size=20),
    Sigmoid(name="sigmoid_1"),
    Dense(input_size=20, output_size=2,name="dense_2"),
    BatchNormalization(name="bn_2",input_size=2),
    Softmax(name="softmax_1")
])
```
#### 4.1.2 Training the Neural Network
```
train(net, inputs, targets, num_epochs=500,loss=CrossEntropy())
```

#### 4.2 CNN

#### 4.1 The network structure
```
net = NeuralNet([
                    Convolution_2D(name="conv_1", filter_shape=(10,1,1,1),padding=0,stride=1),
                    Avg_Pool_2D(name="avg_pool_1", size=2, stride=2),
                    SpatialBatchNormalization(name="sbn_1",input_channel=10),
                    ReLU(name="relu_1"),
                    Convolution_2D(name="conv_2", filter_shape=(20,10,3,3),padding=1,stride=1),
                    Avg_Pool_2D(name="avg_pool_2", size=2, stride=2),
                    SpatialBatchNormalization(name="sbn_2",input_channel=20),
                    ReLU(name="relu_2"),
                    Flatten(name="flat_1"),
                    Dense(input_size=7*7*20, output_size=100, name="dense_1"),
                    BatchNormalization(name="bn_1",input_size=100),
                    ReLU(name="relu_3"),
                    Dense(input_size=100, output_size=10, name="dense_2"),
                    BatchNormalization(name="bn_2",input_size=10),
                    Softmax(name="softmax_1")
                 
                 
                ])
```



## References
* [Softmax Function](https://www.dropbox.com/s/rxrtz3auu845fuy/Softmax.pdf?dl=0)
* [Differentiation of Cross Entropy with Softmax Function](https://stats.stackexchange.com/questions/277203/differentiation-of-cross-entropy)
* [Introduction of Dropout](https://blog.csdn.net/stdcoutzyx/article/details/49022443)
* [Implementation of Dropout](https://blog.csdn.net/hjimce/article/details/50413257)
* [Introduction of Different Optimizers](https://blog.csdn.net/u010089444/article/details/76725843)
* [Implementation of Different Optimizers](https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/)
* [Implementation of Forward and Backward of BN](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
* [Difference between Training and Testing of BN Layer](https://www.quora.com/How-does-batch-normalization-behave-differently-at-training-time-and-test-time)
* [Stanford CS231n](http://cs231n.github.io/convolutional-networks/)


