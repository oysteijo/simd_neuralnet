import numpy as np
from collections import namedtuple

Layer = namedtuple("Layer", ["weight", "bias", "activation_func"])
sigmoid = lambda x: 1.0 / (np.exp( -x ) + 1.0)

class NeuralNet(object):
    def __init__(self, sizes):
        self.n_layers = len(sizes) - 1
        self.layers = [Layer(np.random.randn( inp, out ).astype(np.float32), np.zeros( out, dtype=np.float32 ), sigmoid )
                for inp, out in zip(sizes[:-1], sizes[1:])]
    def feedforward(self, x):
        print(x)
        for layer in self.layers:
            x = layer.activation_func( np.dot( x ,layer.weight) + layer.bias )
            print(x)
        return x

    def backpropagation( self, x, y):
        pass

    def get_weights( self ):
        retlist = []
        for layer in self.layers:
            retlist.append( layer.weight )
            retlist.append( layer.bias )
        return tuple(retlist)

if __name__ == '__main__':
    np.random.seed(42)

    nn = NeuralNet([32,16,8,4])
    print(nn)
    np.savez("initial_weights.npz", *nn.get_weights())
    inp = np.random.rand(32).astype(np.float32)
    np.save("random_input.npy", inp)
    print(nn.feedforward(inp))


