import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import keras.backend as K
import numpy as np

# FIXME: Clean upt this such that it can be initialized the same way as the C code 
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
    optimizer = optimizers.SGD( lr=0.01, momentum=0.9, nesterov=True )
    #optimizer = optimizers.Nadam()
    model.compile( optimizer=optimizer, metrics=["mse", "acc"], loss='binary_crossentropy' )
    return model
    
epochs = 10
nn = build_model()
arr = np.load("mushroom_train.npz")
train_X, train_Y, test_X, test_Y = tuple(arr[x] for x in arr.files)

init_weights = np.load("initial_mushroom_111_32_1.npz")
init_weights_tuple = tuple( init_weights[x] for x in init_weights.files)
nn.set_weights( init_weights_tuple )
nn.fit(x=train_X, y=train_Y, batch_size=16, epochs=epochs, validation_data=(test_X, test_Y), verbose=2, shuffle=False )

keras_weights = nn.get_weights()

# Now we run the C code
from subprocess import run
run( ["./test_sgd", "--learning_rate=%f" % K.eval(nn.optimizer.lr),
        "--momentum=%f" % K.eval(nn.optimizer.momentum),
        "--nesterov=%s" % "true" if nn.optimizer.nesterov else "false"] )

arr = np.load("after-%d-epochs.npz" % epochs )
dobos_weights = tuple( arr[x] for x in arr.files )

tot_err = 0.0
n_params = 0
for k, d in zip(keras_weights, dobos_weights ):
    tot_err += np.absolute( k - d ).sum()
    n_params += k.size

print( "MAE: ", tot_err / n_params )

