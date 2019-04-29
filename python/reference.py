import sys
import numpy as np
from recordclass import recordclass

Layer = recordclass("Layer", ["weight", "bias", "activation_func", "activation_derivative"])

# This corresponds to activation.c in the C implementation
def get_activation_func(str):
    return getattr(sys.modules[__name__], str)

sigmoid = lambda x: 1.0 / (np.exp( -x ) + 1.0)
softmax = lambda x: np.exp(x) / np.exp(x).sum()
relu = lambda x: np.maximum( x, 0, x)
linear = lambda x: x
tanh = np.tanh
exponential = np.exp
softplus = lambda x: np.log1p( np.exp(x) ) 
softsign = lambda x: x / (np.absolute(x) + 1.0 )
hard_sigmoid = lambda x: np.clip((0.2 * x) + 0.5, 0, 1, out=x) # Check this!


sigmoid_derivative = lambda x:  x * (1 - x )
softmax_derivative = lambda x: x 
relu_derivative = lambda x: something where  np.maximum( x, 0, x)
linear_derivative = lambda x: x
tanh_derivative = lambda x: 1.0 - x*x
exponential_derivative = np.exp
softplus_derivative = lambda x: (np.exp(x) - 1) / np.exp(x)
def softsign_derivative(x):
    y = x / (1-x)
    return 1.0 / ((1+np.absolute(y)) * (1+np.absolute(y)))

hard_sigmoid_derivative = lambda x: np.clip((0.2 * x) + 0.5, 0, 1, out=x) # Check this!



def binary_crossentropy(x):
    pass
def categorical_crossentropy(x):
    pass



class NeuralNet(object):
    def __init__(self, sizes):
        self.n_layers = len(sizes) - 1
        self.layers = [Layer(np.random.randn( inp, out ).astype(np.float32), np.zeros( out, dtype=np.float32 ), sigmoid )
                for inp, out in zip(sizes[:-1], sizes[1:])]
#        self.layers[-1].activation_func = softmax

    def predict(self, x):
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
        #grad_b[-1] = 2.0 * ( x-y ) / self.layers[-1].weight.shape[1] # derivative of *mean* square error.
        grad_b[-1] = ( x-y ) / self.layers[-1].weight.shape[1]

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
    np.set_printoptions(precision=4, linewidth=220)
    np.random.seed(42)

    sizes=[32,16,8,4]
    nn = NeuralNet(sizes)
    print(nn)
    np.savez("initial_weights.npz", *nn.get_weights())
    inp = np.random.rand(sizes[0]).astype(np.float32)
    np.save("random_input.npy", inp)
    print(nn.feedforward(inp))

    #train_sample = np.array([0.5]*sizes[-1], dtype=np.float32)
    train_sample = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    grads = []
    for grad_w, grad_b in nn.backpropagation( inp, train_sample ):
        grads.append(grad_b)
        grads.append(grad_w)
        
    print("\n".join(map(str,grads)))
#    for _ in range(100):
#        learning_rate = 0.1
#        for ((grad_w, grad_b), l) in zip(nn.backpropagation(inp, train_sample ), nn.layers):
#            l.weight -= learning_rate * grad_w
#            l.bias -= learning_rate * grad_b

#        print(nn.feedforward(inp))

