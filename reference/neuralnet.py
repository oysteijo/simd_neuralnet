import sys
import numpy as np

# This corresponds to activation.c in the C implementation
activations_implemented = [
    "sigmoid",
    "softmax",
    "relu",
    "linear",
    "tanh",
    "exponential",
    "softplus",
    "softsign",
    "hard_sigmoid"
]

def get_activation_func(str):
    return getattr(sys.modules[__name__], str)

sigmoid      = lambda x: 1.0 / (np.exp( -x ) + 1.0)
softmax      = lambda x: np.exp(x-x.max()) / np.exp(x-x.max()).sum()  # Subtracting x.max() for numerical stability.
relu         = lambda x: np.maximum( x, 0, x)
linear       = lambda x: x
tanh         = lambda x: np.tanh
exponential  = lambda x: np.exp
softplus     = lambda x: np.log1p( np.exp(x) ) 
softsign     = lambda x: x / (np.absolute(x) + 1.0 )
hard_sigmoid = lambda x: np.clip((0.2 * x) + 0.5, 0, 1, out=x) # Check this!

# Argh! The lambdas are anonymous, so we have to set the name specifically to get the lookup.
for funcname in activations_implemented:
    get_activation_func(funcname).__name__ = funcname

# Unless we gave them a name, as above, or implemented these as
# real functions, this function would not have worked obviously.
def get_activation_derivative(func):
    return getattr(sys.modules[__name__], func.__name__ + "_derivative")

sigmoid_derivative      = lambda x: x * (1 - x )
softmax_derivative      = lambda x: 1 
relu_derivative         = lambda x: np.where( x <= 0.0, 0.0, 1.0 )
linear_derivative       = lambda x: 1
tanh_derivative         = lambda x: 1.0 - x*x
exponential_derivative  = lambda x: x
softplus_derivative     = lambda x: (np.exp(x) - 1) / np.exp(x)
softsign_derivative     = lambda x: (x - np.sign(x))**2    # That math was cool!
hard_sigmoid_derivative = lambda x: np.where( np.logical_and( x <= 1.0 , x >= 0.0 ), 0.2, 0 )

# This corresponds to loss.c in the C implementation
def get_loss_func(str):
    return getattr(sys.modules[__name__], str)

mean_squared_error             = lambda y_pred, y_real: 2.0 * ( y_pred-y_real ) / y_pred.shape[0]
mean_absolute_error            = lambda y_pred, y_real: np.where( y_pred >= y_real, 1.0, -1.0 ) / y_pred.shape[0]
mean_absolute_percentage_error = lambda y_pred, y_real: np.where( y_pred >= y_real, 100.0, -100.0 ) / (np.maximum( y_real, 1e-7, y_real ) * y_real.shape[0] )
categorical_crossentropy       = lambda y_pred, y_real: (y_pred - y_real) 
binary_crossentropy            = lambda y_pred, y_real: (y_pred - y_real) / y_pred.shape[0]

# This is the Layer class.
# It only holds the weight matrix, the bias vectors and the activation func + derivatives. No methods.
# This actually used to be a recordclass, but to avoid that dependency, it is now proper classed.

class Layer(object):
    def __init__(self, weight, bias, activation_func, activation_derivative=None ):
        self.weight = weight
        self.bias   = bias
        if isinstance( activation_func, str ):
            self.activation_func = get_activation_func(activation_func)
        else:
            self.activation_func = activation_func

        if activation_derivative == None:
            self.activation_derivative = get_activation_derivative( self.activation_func )
        elif isinstance( activation_derivative, str ):
            self.activation_derivative = get_activation_func( activation_derivative )
        else:
            self.activation_derivative = activation_derivative

# And this is the neural network itself.
class NeuralNet(object):
    def __init__( self, weights, activations=None, loss=None, **kwargs ):
        if isinstance(weights,str):
            arr = np.load(weights)
            weights = tuple((arr[m] for m in arr if arr[m].dtype==np.float32))
            try:
                activations = [o.decode("ascii") for o in arr['activations']]
            except:
                pass

        if isinstance(activations,str):
            activations = [a.strip() for a in activations.split(",")]

        assert 2*len(activations) == len(weights)
        self.layers = [Layer(w,b,get_activation_func(act))
              for w,b,act in zip( weights[::2], weights[1::2], activations )]

        if isinstance(loss,str):
            loss = get_loss_func(loss)
        self.loss = loss

        # Here comes the "matching" logic
        do_nothing = linear_derivative
        if self.loss == binary_crossentropy:
            if self.layers[-1].activation_func == sigmoid:  
                self.layers[-1].activation_derivative = do_nothing
            else:
                print("Warning: Using 'binary_crossentropy' loss function when output activation is not 'sigmoid'.\n")

        if self.loss == categorical_crossentropy:
            if self.layers[-1].activation_func == softmax:
                self.layers[-1].activation_derivative = do_nothing
            else:
                print("Warning: Using 'categorical_crossentropy' loss function when output activation is not 'softmax'.\n")

    def __len__( self ):
        return len(self.layers)

    def predict(self, x):
        for layer in self.layers:
            x = layer.activation_func( np.dot( x ,layer.weight) + layer.bias )
        return x

    def backpropagation( self, x, y):
        assert self.loss is not None
        grad_w = [np.zeros( l.weight.shape, dtype=np.float32 ) for l in self.layers]
        grad_b = [np.zeros( l.bias.shape, dtype=np.float32 ) for l in self.layers]
        activations = [x]

        # Forward
        for layer in self.layers:
            x = layer.activation_func( np.dot( x ,layer.weight) + layer.bias )
            activations.append(x)

        # Backward pass
        assert self.layers[-1].weight.shape == grad_w[-1].shape
        grad_b[-1] = self.loss( x, y )

        for l in range(1, len(self.layers)+1):            
            if l > 1:
                grad_b[-l] = np.dot(self.layers[-l+1].weight, grad_b[-l+1])
            grad_b[-l] *= self.layers[-l].activation_derivative( activations[-l] )

            grad_w[-l] = np.outer(activations[-l-1],grad_b[-l])
            assert self.layers[-l].weight.shape == grad_w[-l].shape
        return zip( grad_w, grad_b ) 

    def get_weights( self ):
        retlist = []
        for layer in self.layers:
            retlist.append( layer.weight )
            retlist.append( layer.bias )
        return tuple(retlist)

    def save( self, filename ):
        f = dict()
        for i, layer in enumerate(self.layers):
            f["weight_{}".format(i)] = layer.weight
            f["bias_{}".format(i)] = layer.bias
                    
        f["activations"] = np.array([ l.activation_func.__name__ for l in self.layers ]).astype('S') 
        np.savez( filename, **f )
