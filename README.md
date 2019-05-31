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

## Limitations
To be able to achieve the above, we need to set some limitations.

 * **`float32` precision only!** The code will use SIMD instructions, so double precision will slow things down, and float16 is not precise enough and has limited support.
 * **Stochastic training only!** No fancy on the training side. No batch or mini-batches or fancy optimizers.
   (*Update* Well... I have to admit that after some coding I now have stochastic and batch training support, an I have
    implemented SGD with momentum and Nesterov, and AdaGrad and RMSprop and Adam... so this statment is really not valid anymore)
 * **Fully connect feed forward neural networks only!** No support for LSTM, convolutional layers, RNN or whatever.
 * **Limited number of additional features!** No weight initializations methods, etc. (*Update:* I do have an initializing tool that can initialize with xavier (aka glorot_uniform) or kaiming (aka he_normal).  
 
### Loss functions implemented
The following loss functions are implemented.
  * mean_squared_error
  * mean_absolute_error
  * mean_absolute_percentage_error
  * binary_crossentropy
  * categorical_crossentropy
  
### Activations functions supported
The following loss functions are implemented.
  * sigmoid
  * tanh
  * softmax
  * softsign
  * softplus
  * hard_sigmoid
  * relu
  * linear
  * exponential
  
An other limitation might be that the number of units in a hidden layer may have to be a multiple of the number of floats
you can fit in a SIMD register. With SSE this number is 4. With AVX this number is 8. With AVX-512 the number will be 16. There must (of course) be possible to have an arbitrary number of inputs and outputs. 

## Status today (31st May 2019)
I got o a lot of things working now. I actually am really happy about this tools now. I need some cleanup here and there. See the issues.

## Status today (2nd May 2019)
Yes! I strongly believe that Keras, the Python reference and my C implementation now get the same gradients.
I still see that my simple SGD application does not give the error rates I expected, so I will try to compare,
with Keras and other tools to check if it's something wrong with my code or with my expectations.

## Status today (24th April 2019)
The reference implementation in python seems to calculate the exact same values for the gradient as a corresponding model in Keras.
Also, the C implementation seems to calculate the exact same gradient values as the reference. Also the activation functions
has been implemented. The gradient calculation has not been tested for many network configurations.

## Status today (18th April 2019)
The reference implementation in python seems to calculate the exact same values for the gradient as a corresponding model in Keras.

