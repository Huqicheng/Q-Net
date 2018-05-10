from deeplearning.tensor import Tensor
import numpy as np


from deeplearning.layers import BatchNormalization

layer = BatchNormalization(name='bn',input_size=4)


input = np.array([[1,1,3,3],[1,1,11,11]])


print(layer.forward(input,training=True))
