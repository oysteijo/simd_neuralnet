## A tool to create a new neural network

The idea is to keep the neural network code as small and mainteinable as possible.
To achieve this the initialization of new neural networks are done in a separate tool.

The code in this directory will build that tool. It is pure C. (ANSI C99). Currently
the only initializers available is Xavier (aka Glorot) and Kaiming (aka He).

So, Xavier is the best initializer for layers with symmetric activation functions
like `sigmoid`, `tanh`, `softmax`, `hard_sigmoid`, `softsign`.

Kaiming initialization performs best with non-symmetric activation layers like `relu`
and `softplus`.

PLEASE NOTE THAT THIS CODE IS NOT PRODUCTION QUALTY. Expect bugs. 

### Usage

    ./create_initial_weights <outfile.npz> n1 n2 n3 n4 ...  <xavier|kaiming> <xavier|kaiming> ...

Example:

    ./create_initial_weights new_weights.npz 128 64 8 2 kaiming kaiming xavier

The command above will create a new weight file called `new_weights.npz` with 128 inputs,
two layers of 64 nodes and 8 nodes, and an output of 2 units. The layesr will be initialized
with kaiming, kaiming and then xavier for the output.

*Warning:* There is no checking in the code, so this will crash if you put in something insane.

