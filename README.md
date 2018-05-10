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

* Building a Neural Network with one hidden layer and one softmax layer.
net = NeuralNet([
                 Dense(input_size=2, output_size=20),
                 Sigmoid(),
                 Dense(input_size=20, output_size=3),
                 Dropout(0.5),
                 Softmax()
                 ])<br>
* Training the Neural Network
train(net, inputs, targets, num_epochs=1000,loss=CrossEntropy())


