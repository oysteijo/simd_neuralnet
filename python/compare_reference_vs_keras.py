import numpy as np
import reference as ref
import kerasmodel as krs

# split the script here!
with open("config.txt") as f:
    config = {k.strip(): v.strip() for k,v in [x.split("=") for x in f.readlines()]}

# Create two networks
refnn = ref.NeuralNet( **config)
krsnn = krs.create_keras_model( **config )

# Check prediction
inp = np.load(config["test_sample"])
keras_pred = krsnn.predict(inp)
refer_pred = refnn.predict(inp)

print("Mean Absolute error of prediction: ", np.mean( np.absolute(keras_pred - refer_pred)))

# Backprop
target = np.load(config["test_target"])

keras_grad = krs.get_weight_grad(krsnn, inp, target.reshape(1,-1))
refer_grad = refnn.backpropagation( inp.reshape((-1)), target.reshape((-1)))
r_grads = []
for grad_w, grad_b in refer_grad:
    r_grads.append(grad_w)
    r_grads.append(grad_b)

for refg,krsg in zip(r_grads, keras_grad):
    print("Mean Absolute Error of gradient: ", np.mean( np.absolute(refg - krsg)))

