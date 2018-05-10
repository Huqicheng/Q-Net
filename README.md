DeepLearningFramework
=====
A simplified deep learning framework.<br>

## 1. Layers
* Dense<br>
* Dropout<br>
* Activations<br>
  * Sigmoid
  * Tanh
  * Softmax
  * ReLU

  

## 2. Loss Functions
* MSE<br>
* CrossEntropy<br>

## 3. Optimizers
* SGD <br>

## 4. Examples

#### 4.1 xor
The source code of training a Neural Network to fit the xor function is in xor.py.
#### 4.1.1 Building a Neural Network with one hidden layer and one softmax layer.
net = NeuralNet([<br>
                 Dense(input_size=2, output_size=20),<br>
                 Sigmoid(),<br>
                 Dense(input_size=20, output_size=3),<br>
                 Dropout(0.5),<br>
                 Softmax()<br>
                 ])<br>
#### 4.1.2 Training the Neural Network
train(net, inputs, targets, num_epochs=1000,loss=CrossEntropy())


