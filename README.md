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

An other limitation might be that the number of units in a hidden layer may have to be a multiple of the number of floats
you can fit in a SIMD register. With SSE this number is 4. With AVX this number is 8. With AVX-512 the number will be 16. There must (of course) be possible to have an arbitrary number of inputs and outputs. (Comment Feb. 2020: Is this still true?)

