from deeplearning.tensor import Tensor
import numpy as np


from deeplearning.layers import Dropout

layer = Dropout()


input = np.array([[1,2,3,4],[2,3,4,5]])


print(layer.forward(input,training=True))
