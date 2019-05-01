import numpy as np
import reference as ref
import itertools
import subprocess

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
    config["activations"] = "sigmoid,sigmoid,%s" % a
    print("Loss function: ", l)
    print("Output activation: ", a)

    # Create two networks
    refnn = ref.NeuralNet( **config)

    # Do the dirty stuff in C and store it in npy files.
    subprocess.run(["./c_checker", config["weights"], config["activations"], config["loss"], config["test_sample"], config["test_target"]] )
    
    # Check prediction
    inp = np.load(config["test_sample"])
    refer_pred = refnn.predict(inp)

    c_pred = np.load("c_prediction.npy")    

    print("Mean Absolute error of prediction: ", np.mean( np.absolute(c_pred - refer_pred)))
    
    # Backprop
    target = np.load(config["test_target"])
    
    refer_grad = refnn.backpropagation( inp.reshape((-1)), target.reshape((-1)))
    r_grads = []
    for grad_w, grad_b in refer_grad:
        r_grads.append(grad_w)
        r_grads.append(grad_b)
    
    c_grad = []
    for l in range(3):
        c_grad.append( np.load("weight_grad_%d.npy" % l ) )
        c_grad.append( np.load("bias_grad_%d.npy" % l ) )

    for refg,cg in zip(r_grads, c_grad):
        print("Mean Absolute Error of gradient: ", np.mean( np.absolute(refg - cg)))
    np.set_printoptions( precision=4, linewidth= 240)
    print("\n".join(map(str, r_grads)))
    print("\n".join(map(str, c_grad)))
    exit()

