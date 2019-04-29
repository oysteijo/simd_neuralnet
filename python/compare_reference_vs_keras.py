import numpy as np
import reference as ref
import kerasmodel as krs
import itertools
# split the script here!
with open("config.txt") as f:
    config = {k.strip(): v.strip() for k,v in [x.split("=") for x in f.readlines()]}

activations = ["sigmoid", "relu", "linear", "tanh", "softplus", "softsign", "hard_sigmoid", "softmax"]
losses = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error","binary_crossentropy", "categorical_crossentropy"]

for l,a in itertools.product(losses, activations ):
    if l == "binary_crossentropy" and a != "sigmoid":
        continue
    # When it comes to softmax and categorical_crossentropy, you cannot have the one without the other.
    if l == "categorical_crossentropy" and a != "softmax":
        continue
    if a == "softmax" and l != "categorical_crossentropy":
        continue
    
    config["loss"] = l
    config["activations"] = "sigmoid, sigmoid, %s" % a
    print("Loss function: ", l)
    print("Output activation: ", a)

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
#    np.set_printoptions( precision=4, linewidth= 240)
#    print("\n".join(map(str, r_grads)))
#    print("\n".join(map(str, keras_grad)))

