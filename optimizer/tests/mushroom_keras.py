import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np

def build_model( hidden_layer_sizes=(32,) ):
    model = Sequential()
    is_input = True
    for s in hidden_layer_sizes:
        if is_input:
            model.add( Dense( s, activation='relu', input_dim=111 ) )
            is_input = False
        else:
            model.add( Dense( s, activation="relu" ))
            #model.add( Dropout(0.3) )
    model.add( Dense(1, activation='sigmoid'))
    optimizer = optimizers.SGD() # lr=0.01, momentum=0.9, nesterov=True)
    #optimizer = optimizers.Nadam()
    model.compile( optimizer=optimizer, metrics=["mse", "acc"], loss='binary_crossentropy' )
    return model
    
epochs = 100
nn = build_model()
arr = np.load("mushroom_train.npz")
train_X, train_Y, test_X, test_Y = tuple(arr[x] for x in arr.files)

init_weights = np.load("inital_mushrom_111_32_1.npz")
init_weights_tuple = tuple( init_weights[x] for x in init_weights.files)
nn.set_weights( init_weights_tuple )
nn.fit(x=train_X, y=train_Y, batch_size=8, epochs=epochs, validation_data=(test_X, test_Y), verbose=2 )

