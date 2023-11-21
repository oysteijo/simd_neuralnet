# Implementing custom activation functions.

Here is an example of how to implement and use a custom implemented activation function.
This example will implement a simple sigmoid function with a pre scale of the input.

$$ f(x) = \frac{1}{1+e^{-\alpha x}} $$

where $\alpha$ is a fixed constant, ie. $\alpha$ is not a trainable parameter.

    #include <math.h>
    #ifndef SIGMOID_SCALAR
    #define SIGMOID_SCALAR 0.1
    #endif
    
    void scaled_sigmoid( const int n, float *y )
    {
        for (int i = 0; i < n; i++) 
            y[i] = 1.0f / (1.0f + expf(-SIGMOID_SCALAR*y[i]));
    }
    
    void scaled_sigmoid_derivative( const int n, const float *activation, float *d )
    {
        for(int i=0; i < n; i++ )
            d[i] *= SIGMOID_SCALAR*activation[i]*(1.0f-activation[i]);
    }

This is the code that goes into a source file. The file is called `gnubg_activation.c` since this activation function is
used in the GNU Backgammon (gnubg) neural networks. This implementation in this example is not optimized for performance or
accuracy for the original gnubg implementation, so this serves more or less just as an example.

Note the following: 
 - Function signatures of the two functions. This is the standard where n is the number of elements in the vector.
 - The derivative functions takes in the activation as input, not the x values! The derivative function should hence
   calculate the derivative expressed by the activation itself.

Compile the code with the following:

    gcc -O3 -Wall -Wextra -fPIC -shared -c gnubg_activation.c
    gcc -shared gnubg_activation.o -o libgnubg_activation.so -lm

You should now be able to specify the custom activation in a neural network like this:

    neuralnet_t *nn = neuralnet_create( 2,                                                       // n_layers
                                    INT_ARRAY( 3, 4, 2 ),                                        // sizes
                                    STR_ARRAY( "tanh", "scaled_sigmoid@libgnubg_activation.so" ) // activation functions
                                    );

When setting a loss function with `neuralnet_set_loss()` the derivative will be set correctly if it
is implemented and present in the same shared object file as the forward activation.

