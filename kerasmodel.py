# coding: utf-8
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add( Dense( 16, activation='sigmoid', input_dim=32 ) )
model.add( Dense( 8, activation='sigmoid') )
model.add( Dense( 4, activation='sigmoid') )
model.compile( optimizer='SGD', loss="mean_squared_error")

arr = np.load("initial_weights.npz")
inp = np.load("random_input.npy")
inp = inp.reshape(1,32)
    
weights = (arr[m] for m in arr)
model.set_weights(list(weights))


print(model.predict(inp))
