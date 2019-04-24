# simd_neuralnet
Feed-forward neural network implementation in C with SIMD instructions.

## The idea.
At current this is just a study project for myself to improve my abilities to implement a 
feed forward neural network in C. A lot of this code will be based on code from some of my
other projects. Hopefully this will be generally usable.

One of the ideas here is to make sure that this implementation is correct and gives the same
results as other neural network implementations. We will therefore also implement a Python
neural network in the verification. The weights and other parameters of other neural network
libraries should therefore be interchangeable with the weights of these.

Say you train a neural network from Keras, and then you want to use it in your C coded application?
Then you can simply store the weights from your Keras neural network and loed them into this
codes, and you whole neural network is totally independent of both Python and Keras (and a GPU).

## Limitations
To be able to achive the above, we need to set some limitations.

 * **`float32` precision only!** The code will use SIMD instructions, so double precision will slow things down, and float16 is not precise enough and has limited support.
 * **Stochastic training only!** No fancy on the training side. No batch or mini-batches or fancy optimizers.
 * **Fully connect feed forward neural networks only!** No support for LSTM, convolutional layers, RNN or whatever.
 * **Limited number of additional features!** No weight initializations methods, etc.
 
An other limitation might be that the number of units in a hidden layer may have to be a multiple of the number of floats
you can fit in a SIMD register. With SSE this number is 4. With AVX this number is 8. With AVX-512 the number will be 16. There must (of course) be possible to have an abritary number of inputs and outputs. 

## Status today (24th April 2019)
The reference implementation in python seems to calculate the exact same values for the gradient as a corresponding model in Keras.
Also, the C implementation seems to calculate the exact same gradient values as the reference. Also the activation functions
has been implemented. The gradient calculation has not been tested for many network configurations.

## Status today (18th April 2019)
The reference implementation in python seems to calculate the exact same values for the gradient as a corresponding model in Keras.

