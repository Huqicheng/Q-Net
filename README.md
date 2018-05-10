DeepLearningFramework
=====
A simplified deep learning framework.<br>

## Implemented Layers
* Dense<br>
* Dropout<br>
* Activations<br>
  * Sigmoid
  * Tanh
  * Softmax
  * ReLU

  

## Implemented Loss Functions
* MSE<br>
* CrossEntropy<br>

## An Example

### xor
#### Building a Neural Network with one hidden layer and one softmax layer.
net = NeuralNet([<br>
                 Dense(input_size=2, output_size=20),<br>
                 Sigmoid(),<br>
                 Dense(input_size=20, output_size=3),<br>
                 Dropout(0.5),<br>
                 Softmax()<br>
                 ])<br>
#### Training the Neural Network
train(net, inputs, targets, num_epochs=1000,loss=CrossEntropy())


