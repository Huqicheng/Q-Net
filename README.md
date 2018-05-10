DeepLearningFramework
=====
A simplified deep learning framework.<br>

## 1. Implemented Layers
* Dense<br>
* Dropout<br>
* Activations<br>
  * Sigmoid
  * Tanh
  * Softmax
  * ReLU

  

## 2. Implemented Loss Functions
* MSE<br>
* CrossEntropy<br>

## 3. Examples

### 3.1 xor
#### 3.1.1 Building a Neural Network with one hidden layer and one softmax layer.
net = NeuralNet([<br>
                 Dense(input_size=2, output_size=20),<br>
                 Sigmoid(),<br>
                 Dense(input_size=20, output_size=3),<br>
                 Dropout(0.5),<br>
                 Softmax()<br>
                 ])<br>
#### 3.1.2 Training the Neural Network
train(net, inputs, targets, num_epochs=1000,loss=CrossEntropy())


