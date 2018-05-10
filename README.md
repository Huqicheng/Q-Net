DeepLearningFramework
=====
A simplified deep learning framework.<br>

## 1. Layers
* Dense
* Dropout
* Activations
  * Sigmoid
  * Tanh
  * Softmax
  * ReLU
* BatchNormalization

  

## 2. Loss Functions
* MSE
* CrossEntropy

## 3. Optimizers
* SGD 
* Momemtum_SGD 

## 4. Examples

#### 4.1 xor
The source code of training a Neural Network to fit the xor function is in xor.py.
#### 4.1.1 Building a Neural Network with one hidden layer and one softmax layer.
```
net = NeuralNet([
    Dense(input_size=2, output_size=20),
    Sigmoid(),
    Dense(input_size=20, output_size=2),
    Dropout(0.5),
    Softmax()
 ])
```
#### 4.1.2 Training the Neural Network
```
train(net, inputs, targets, num_epochs=1000,loss=CrossEntropy())
```

## References
* [Softmax Function](https://www.dropbox.com/s/rxrtz3auu845fuy/Softmax.pdf?dl=0)
* [Differentiation of Cross Entropy with Softmax Function](https://stats.stackexchange.com/questions/277203/differentiation-of-cross-entropy)
* [Introduction of Dropout](https://blog.csdn.net/u010089444/article/details/76725843)
* [Implementation of Dropout](https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/)
* [Implementation of forward and backward of BN](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
* [Difference between training and testing process of BN Layer](https://www.quora.com/How-does-batch-normalization-behave-differently-at-training-time-and-test-time)


