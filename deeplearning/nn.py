from typing import Sequence, Iterator, Tuple

from deeplearning.tensor import Tensor
from deeplearning.layers import Layer


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
    
    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        training = kwargs['training']
        for layer in self.layers:
            inputs = layer.forward(inputs,training=training)
        return inputs

    def predict(self, inputs: Tensor) -> Tensor:
        output = self.forward(inputs, training=False)
        y_pred = output.argmax(axis=1)
        return y_pred

    def predict_prob(self, inputs: Tensor) -> Tensor:
        output = self.forward(inputs, training=False)
        y_pred = output.argmax(axis=1)
        y_pred_prob = output[0][y_pred[0]]
        return y_pred_prob

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield layer.name+'_'+name, param, grad


