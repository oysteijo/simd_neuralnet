# coding: utf-8
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import keras.backend as K 

model = Sequential()
model.add( Dense( 16, activation='sigmoid', input_dim=32 ) )
model.add( Dense( 8, activation='sigmoid') )
model.add( Dense( 4, activation='sigmoid') )
model.compile( optimizer=SGD(lr=0.1), loss="binary_crossentropy")

arr = np.load("initial_weights.npz")
inp = np.load("random_input.npy")
inp = inp.reshape(1,32)
    
weights = (arr[m] for m in arr)
model.set_weights(list(weights))


print(model.predict(inp))

train_sample = np.array([1.0,1.0,0.0,0.0], dtype=np.float32)
train_sample = train_sample.reshape(1,4)
#model.train_on_batch(inp, train_sample )

print(model.predict(inp))

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

np.set_printoptions(precision=4, linewidth=130) 
print(get_weight_grad(model, inp, train_sample))
