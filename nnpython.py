import numpy as np
from recordclass import recordclass

Layer = recordclass("Layer", ["weight", "bias", "activation_func"])
sigmoid = lambda x: 1.0 / (np.exp( -x ) + 1.0)

class NeuralNet(object):
    def __init__(self, sizes):
        self.n_layers = len(sizes) - 1
        self.layers = [Layer(np.random.randn( inp, out ).astype(np.float32), np.zeros( out, dtype=np.float32 ), sigmoid )
                for inp, out in zip(sizes[:-1], sizes[1:])]
    def feedforward(self, x):
        for layer in self.layers:
            x = layer.activation_func( np.dot( x ,layer.weight) + layer.bias )
        return x

    def backpropagation( self, x, y):
        nabla_w = [np.zeros( l.weight.shape, dtype=np.float32 ) for l in self.layers]
        nabla_b = [np.zeros( l.bias.shape, dtype=np.float32 ) for l in self.layers]
        activations = [x]
        # Forward
        for layer in self.layers:
            x = layer.activation_func( np.dot( x ,layer.weight) + layer.bias )
            activations.append(x)
        # Backward pass
        #delta = (x-y)*y*(1-y)  # Derivative of the cost * derivative of sigmoid 
        #        nabla_b[-1] = delta
        #       nabla_w[-1] = np.outer(activations[-2],delta)
        assert self.layers[-1].weight.shape == nabla_w[-1].shape
        error = x-y
        for l in range(1, len(self.layers)+1):            
            if l == 1:
                delta = error # *y*(1-y)  # Derivative of the cost * derivative of sigmoid 
            else:
                delta = np.dot(self.layers[-l+1].weight, delta) #* (activations[-l]*(1.0-activations[-l]))

            delta *= activations[-l]*(1.0-activations[-l])
            
            assert nabla_b[-l].shape == delta.shape
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(activations[-l-1],delta)
            assert self.layers[-l].weight.shape == nabla_w[-l].shape
        return zip( nabla_w, nabla_b ) 

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

    train_sample = np.array([0.5,0.5,0.5,0.5], dtype=np.float32)

    for _ in range(100):
        learning_rate = 0.1
        for ((n_w, n_b), l) in zip(nn.backpropagation(inp, train_sample ), nn.layers):
            l.weight -= learning_rate * n_w
            l.bias -= learning_rate * n_b

        print(nn.feedforward(inp))




