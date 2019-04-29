# coding: utf-8
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import keras.backend as K 

def create_keras_model( weights, activations, loss, **kwargs ):

    arr = np.load(weights)
    weights = tuple((arr[m] for m in arr))
    activations = [a.strip() for a in activations.split(",")]

    assert 2*len(activations) == len(weights)

    model = Sequential()
    model.add( Dense( weights[0].shape[1], activation=activations[0], input_dim=weights[0].shape[0] ) )
    for w, act in zip( weights[2::2], activations[1:] ):
        model.add( Dense( w.shape[1], activation=act) )
    model.compile( optimizer=SGD(lr=0.01), loss=loss)
    model.set_weights(list(weights))
    return model

# This function is taken from a Stack Overflow posting.
# https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad
    
    print(get_weight_grad(model, inp, train_sample))

if __name__ == "__main__":
    with open("config.txt") as f:
        config = {k.strip(): v.strip() for k,v in [x.split("=") for x in f.readlines()]}
    # print(config)
    model = create_keras_model( **config )

    inp = np.load( config["test_sample"])
    inp = inp.reshape(-1,model.get_layer(index=0).input_shape[1])
    target =  np.load(config["test_target"])
    target = target.reshape(-1, model.getlayer(index=-1).output_shape[1])



    print( inp )        
