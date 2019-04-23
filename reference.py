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
        grad_w = [np.zeros( l.weight.shape, dtype=np.float32 ) for l in self.layers]
        grad_b = [np.zeros( l.bias.shape, dtype=np.float32 ) for l in self.layers]
        activations = [x]

        # Forward
        for layer in self.layers:
            x = layer.activation_func( np.dot( x ,layer.weight) + layer.bias )
            activations.append(x)

        # Backward pass
        assert self.layers[-1].weight.shape == grad_w[-1].shape
        grad_b[-1] = 2.0 * ( x-y ) / self.layers[-1].weight.shape[1] # derivative of root *mean* square error.

        for l in range(1, len(self.layers)+1):            
            if l > 1:
                grad_b[-l] = np.dot(self.layers[-l+1].weight, grad_b[-l+1])
            grad_b[-l] *= activations[-l]*(1.0-activations[-l])  # FIXME: This is depending on the activation_func
            
            grad_w[-l] = np.outer(activations[-l-1],grad_b[-l])
            assert self.layers[-l].weight.shape == grad_w[-l].shape
        return zip( grad_w, grad_b ) 

    def get_weights( self ):
        retlist = []
        for layer in self.layers:
            retlist.append( layer.weight )
            retlist.append( layer.bias )
        return tuple(retlist)

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=130)
    np.random.seed(42)

    sizes=[32,16,8,4]
    nn = NeuralNet(sizes)
    print(nn)
    np.savez("initial_weights.npz", *nn.get_weights())
    inp = np.random.rand(sizes[0]).astype(np.float32)
    np.save("random_input.npy", inp)
    print(nn.feedforward(inp))

    train_sample = np.array([0.5]*sizes[-1], dtype=np.float32)

    grads = []
    for grad_w, grad_b in nn.backpropagation( inp, train_sample ):
        grads.append(grad_w)
        grads.append(grad_b)
        
    print(grads)
#    for _ in range(100):
#        learning_rate = 0.1
#        for ((grad_w, grad_b), l) in zip(nn.backpropagation(inp, train_sample ), nn.layers):
#            l.weight -= learning_rate * grad_w
#            l.bias -= learning_rate * grad_b

#        print(nn.feedforward(inp))

