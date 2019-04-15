# Example. Make a new neural net of sizes 32,16,8,4
# make one forward pass and then one backward.

import numpy as np
import random

from neuralnet import Network

np.random.seed(32)
nn = Network((32,16,8,4 ))

inp = np.random.rand(2,32)
print("input: ", inp)
print(nn.feedforward(inp))

print("input: ", inp)
 

target = np.random.rand(2,4)
print(nn.backprop( inp, target))

