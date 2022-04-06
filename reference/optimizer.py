import numpy as np

class Optimizer(object):

    def __init__(self, nn, batch_size=1, metrics=None, callbacks=None, shuffle=True,  **kwargs ):
        self.nn = nn
        self.iterations = 0
        self.batch_size = batch_size
        self.metrics    = metrics
        self.callbacks  = callbacks
        self.shuffle    = shuffle

    def run_epoch( self, train_X, train_Y, valid_X=None, valid_Y=None ):
        pass

class SGD(Optimizer):

    def __init__( self, nn, learning_rate=0.01, decay=0.0, **kwargs ):
        super( SGD, self).__init__( nn, **kwargs )
        self.learning_rate = learning_rate
        self.decay = decay

    def run_epoch( self, train_X, train_Y, valid_X=None, valid_Y=None ):
        for inp, target in zip(train_X, train_Y):
            grads = self.nn.backpropagation( inp, target )
            for layer, (grad_w, grad_b) in zip( self.nn.layers , grads ):
                layer.weight -= self.learning_rate * grad_w
                layer.bias   -= self.learning_rate * grad_b
        if ( (valid_X is not None ) and (valid_Y is not None) ):
            preds = self.nn.predict( valid_X )
            print(" mean squared error valid: ",  np.square(preds - valid_Y).mean() )

