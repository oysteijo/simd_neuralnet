# simd_neuralnet
Feed-forward neural network implementation in C with SIMD instructions.

BTW: simd_neuralnet is a bit of a misnomer. It is not SIMD instructions apart from AVX and AVX2 instructions in some of the most heavy functions. Can anyone come up with a better name? It is a neural network library written in C. I'm not sure what else makes this library different from the other libraries out there.

## The idea.
At current this is just a study project for myself to improve my abilities to implement a 
feed forward neural network in C. A lot of this code will be based on code from some of my
other projects. Hopefully this will be generally usable.

One of the ideas here is to make sure that this implementation is correct and gives the same
results as other neural network implementations. We will therefore also implement a Python
neural network in the verification. The weights and other parameters of other neural network
libraries should therefore be interchangeable with the weights of these.

Say you train a neural network from Keras, and then you want to use it in your C coded application?
Then you can simply store the weights from your Keras neural network and load them into this
codes, and you whole neural network is totally independent of both Python and Keras (and a GPU).

## What is it good for?
Since this is pure ANSI C this library will be perfect for embedded devices. I'm also using
this library in a online learning system where only one sample is fed into the trainer at a time
in a reinforcement learning pattern. This library then works perfectly since there is no need to
struggle with python bindings or slow memory transfers to GPU memory.

## Limitations
To be able to achieve the above, we need to set some limitations.

 * **`float32` precision only!** The code will use SIMD instructions, so double precision will slow things down, and float16 is not precise enough and has limited support.
 * **Fully connect feed forward neural networks only!** No support for LSTM, convolutional layers, RNN or whatever.
 
### Loss functions implemented
The following loss functions are implemented.
  * mean_squared_error
  * mean_absolute_error
  * mean_absolute_percentage_error
  * binary_crossentropy
  * categorical_crossentropy
  
### Activations functions supported
The following activation functions are implemented.
  * sigmoid
  * tanh
  * softmax
  * softsign
  * softplus
  * hard_sigmoid
  * relu
  * linear
  * exponential

### Parameter initialization methods implemented
  * Xavier (aka Glorot uniform)
  * Kaiming (aka He normal)

### Optimizers implemented
  * SGD (Stochastic Gradient Descent)
  * RMSprop
  * Adagrad
  * Adam
  * AdamW

Most of these optimizers can also handle momentum and Nesterov momentum.

### Metric functions implemented
  * mean_squared_error
  * mean_absolute_error
  * mean_absolute_percentage_error
  * binary_crossentropy
  * categorical_crossentropy
  * binary_accuracy

### Plan ahead
So, the idea is to keep this small and beautiful. Features, like:
  * more activations
  * more loss functions
  * more initialization
  * more optimizers
  * more callbacks

... can be added successivly as needed. Please make pull requests. 

## The file format
The neural networks are stored as numpy arrays in the `.npz` in PKZIP manner. It is therefore simple to
interact between python based neural network frameworks and simd_neuralnet. Maybe you want to train your
neural networks with Keras and the deploy you trained neural network with simd_neuralnet? Not a problem at all.

### Details.
The arrays that makes the weights and biases in the layers of the neural net, are simply stored as 
`.npy` "files" in an PKZIP archive file which is the very same way numpy stores it's arrays in `.npz` files.
They are stores in input to output layer order, with weights first and bias following. To keep track of the
activation functions used, a simple array of strings (actually character arrays) are stored at the end of
the archive. Actually, the activations function string array can be stores anywhere in the chain of other
arrays, however the `neuralnet_save()` function will store it at the en of the zip archive.

A simple feed forward neural network with 3 inputs, 4 hidden units and 2 outputs, and tanh and sigmoid activations
will then be stored as.

    A numpy array with shape 3,4 (Weights)
    A numpy array with shape 4,
    A numpy array with shape 4,2
    A numpy array with shape 2,
    A numpy array of string (char arrays) contaning "tanh" and "sigmoid"

Let me build that neural network in C code:

    #include "neuralnet.h"
    int main(){
        neuralnet_t *nn = neuralnet_create( 2, INT_ARRAY( 3, 4, 2 ), STR_ARRAY( "tanh", "sigmoid"));
        neuralnet_initialize( nn, "xavier", "xavier" );

        /* I'm only showing this for being able to save, so I won't set a loss function */
        neuralnet_save( nn, "my_first_neuralnet_%d_%d_%d.npz", 3,4,2 );
        neuralnet_free( nn );

        return 0;
    }

I am then able to open this file in python and numpy:

    Python 3.8.1 (default, Jan 22 2020, 06:38:00) 
    [GCC 9.2.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import numpy as np
    >>> weights = np.load("my_first_neuralnet_3_4_2.npz")
    >>> params = [ weights[i] for i in weights.files]
    >>> import pprint
    >>> for array in params:
    ...    pprint.pprint( array )
    ... 
    array([[ 0.6299053 , -0.19556482,  0.52419794,  0.5526036 ],
           [ 0.7622228 , -0.56002605, -0.3051082 ,  0.49666473],
           [-0.41148126,  0.09993298, -0.04185252,  0.23862255]],
          dtype=float32)
    array([0., 0., 0., 0.], dtype=float32)
    array([[-0.27043104,  0.02680182],
           [ 0.9044595 ,  0.8323902 ],
           [ 0.27142346,  0.43459392],
           [-0.71679485,  0.21393776]], dtype=float32)
    array([0., 0.], dtype=float32)
    array([b'tanh', b'sigmoid'], dtype='|S7')

And of course it also works the other way, I can create an array in numpy and then stroe it back to
be read in C. I only need to make sure I store the activation names in pure ascii bytes. (No unicode)

    w1 = np.random.random( (3,4) ).astype(np.float32) 
    b1 = np.zeros( (4,), dtype=np.float32 )
    w2 = np.random.random( (4,2) ).astype(np.float32)
    b2 = np.zeros( (2,), dtype=np.float32 )
    activations = np.array(["tanh", "sigmoid"]).astype('S')
    
    np.savez( "numpy_saved_neuralnet.npz", w1, b1, w2, b2, activations )

And that file can be used with `neuralnet_load()`.





