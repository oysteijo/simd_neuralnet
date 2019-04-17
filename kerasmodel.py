# coding: utf-8
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add( Dense( 16, activation='sigmoid', input_dim=32 ) )
model.add( Dense( 8, activation='sigmoid') )
model.add( Dense( 4, activation='sigmoid') )
model.compile( optimizer=SGD(lr=0.1), loss="mean_squared_error")

arr = np.load("initial_weights.npz")
inp = np.load("random_input.npy")
inp = inp.reshape(1,32)
    
weights = (arr[m] for m in arr)
model.set_weights(list(weights))


print(model.predict(inp))

train_sample = np.array([0.5,0.5,0.5,0.5], dtype=np.float32)
train_sample = train_sample.reshape(1,4)
model.train_on_batch(inp, train_sample )

print(model.predict(inp))

