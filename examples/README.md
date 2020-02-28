# A working example

This directory contains code that will serve as an example of how to get started
with this software and explain some of the concepts. 

## A mushroom classifier

At the UCI Machine Learning repository there was a mushroom dataset posted.
The dataset was donated to UCI ML in April back in 1987, but will serve as a good
first example for use of this software.

The dataset contains 8124 samples of different mushrooms where each sample is
given 23 different categorical features. One of the categorical features is
whether the mushroom is edible or poisonous, and in this example we will build
a simple classifier from the other features. The data is provided as a `.csv`
file in this directory.

### Mushroom feature engineering

This software, `simd_neuralnet` is a neural network library. It does not to any
feature engineering or other machine learning tricks. We will therefore use python
and numpy to do simple feature engineering (just one hot encoding). We will also
randomize the order of the samples and then split into a train partition and a
test partition.

Let's start python:

    # First we import the libraries we are using
    import csv
    import numpy as np
    from collections import defaultdict
    
    columns = defaultdict(list) # each value in each column is appended to a list
    
    # Read the file into a dictionary where each column is tehe key and the value
    # is a list of all elements in the column.
    with open('mushrooms.csv') as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                columns[k].append(v) # append the value into the appropriate list
                                     # based on column name k
    
    # This is code does one-hot-encoding of all the features. It finds the number of unique
    # elements in each column and makes a numpy array of one hot encoded features for each.
    # At the end of each loop iterations the numpy array is concatenated to the other
    # previous features form previous iterations. 
    # It there is only one unique value in a column, that column will be discarded in the
    # features. If there are only two unique values in a column, it will only produce one
    # feature column since another column will be complementary to the other.
    
    features = None
    for c in list(columns.keys()):
        u, e =  np.unique(columns[c], return_inverse=True)
        if len(u) == 1:
            continue
        if len(u) == 2:
            f = np.array(e, dtype=np.float32).reshape((-1,1))
        else:
            n_values = np.max(e) + 1
            f = np.eye(n_values)[e].astype(np.float32)
        print( "{:25s}".format(c), f.shape )
        if features is None:
            features = f
        else:
            features = np.hstack((features, f ))
    
    # "features" is actually a bit of a misnomer as the first column is actually the target value.
    # We will also use python to split the dataset. First we randomize the order, and then we split
    # into train and test partitions.
    
    np.random.shuffle( features )
    split_ratio = 0.7
    split_idx = int(features.shape[0] * split_ratio)
    
    # We save everything into four numpy arrays in the `.npz` format (which is actually zip).
    # - train_features
    # - train_targets
    # - test_features
    # - test_targets
    
    np.savez("mushroom_train.npz", features[:split_idx,1:], features[:split_idx,0].reshape((-1,1)),
                                   features[split_idx:,1:], features[split_idx:,0].reshape((-1,1)))
    
The code above is also available in `mushroom_to_numpy.py`.

### Building a simd_neuralnet from scratch in ANSI C.

`simd_neuralnet` is a library written in ANSI C and here comes an example of the usage in C.
Let's start simple.

    #include "npy_array.h"
    #include "neuralnet.h"
    
    #include <stdio.h>
    #include <stdlib.h>
    #include <assert.h>
    ...

First we include the header file from npy_array (`#include "npy_array.h"`) which is a library that
can read and write numpy arrays stored as `.npy` or `.npz` files. The routines for this from this
library will be used to read the training and test data we saved in the python code above.

The next include (`#include "neuralnet.h"`) is the header declarations for the neural network itself.

The three last includes are standard include files that comes from the C library.

    int main( int argc, char *argv[] )
    {
        npy_array_list_t *filelist = npy_array_list_load( "mushroom_train.npz" );
        assert( filelist );
        
        npy_array_list_t *iter = filelist;
        npy_array_t *train_X = iter->array;  iter = iter->next;
        npy_array_t *train_Y = iter->array;  iter = iter->next;
        npy_array_t *test_X = iter->array;   iter = iter->next;
        npy_array_t *test_Y = iter->array;
    
        /* if any of these asserts fails, try to open the weight in python and save with
         * np.ascontiguousarray( matrix ) */
        assert( train_X->fortran_order == false );
        assert( train_Y->fortran_order == false );
        assert( test_X->fortran_order == false );
        assert( test_Y->fortran_order == false );
    ...

The above code opens a main() function which is the starting point of our little program.
The function `npy_array_list_load()` loads a `.npz` file into memory and makes a linked
list of all the numpy arrays it finds in the archive. The variable `filelist` is set to
point to the first item in the list. It is then setting up pointers to the npy_arrays
through the list. As the we expect the file to have four arrays (as we saved then in
the code above) we move an iterator one step in between each assignment.

The hence get assigned:
 * `train_X`  - The train partition features
 * `train_Y`  - The train partition target
 * `test_X`   - The test partition features
 * `test_Y`   - The test partition target

These are all pointers to an numpy array in memory. Read more about the structure in the
npy_array documentation.

There is also some assert macros added to ensure that the row/column order is correct. That
is the `fortran_order` lines. 

    ...
        neuralnet_t *nn = neuralnet_create( 3,
                INT_ARRAY( train_X->shape[1], 64, 32, 1 ),
                STR_ARRAY( "relu", "relu", "sigmoid" ) );
        assert( nn );
    
        neuralnet_initialize( nn, STR_ARRAY("kaiming", "kaiming", "kaiming"));
        neuralnet_set_loss( nn, "binary_crossentropy" );
    ...

These next lines creates a new neural network datatype, and it is followed by an assert
macro call to check that it really got created. The arguments the `nueralnet_create()`
function is first a number indicating how many layers you want in your neural network.
The next argument is an array of integers indicating the sizes of the inputs and outputs
of each layer. The output size of one layer must be equal to the input of the next layer,
So the length of this array must be one more than the number of layers.

We also know that then number of inputs to the neural network must match the number of
features. The number of features is given by the number of columns in the training features,
hence we can pass in `train_X->shape[1]` as the first integer in the list. (For the mushroom
case, I think the number is 111). The next integers indicate the of the next levels. It
is good advice to keep these multiples of the number of floats you can store in an SIMD
register. This is 4 for SSE, 8 for AVX and 16 for AVX-512. The last integer in the
size array is the number of outputs of the neural network. We are doing a binary
classification, so one output (edible or poisonous) should do.

The next inputs to `neuralnet_create()` is an array of strings that sets the activation functions
at the outputs of each layer. I picked "relu" activations for intermediate layers, and then 
"sigmoid" activation on the output, such that the out will approach 0 or 1 depending on which
class it predicts.

That is how to create a `neuralnet_t` datatype. Note that this datatype is already able to perform a
prediction, however as it is not it is just allocated and memory for all parameters has also
been allocated. The weights are just random values, and not even initialized to proper values.
So, yes, it can do a prediction, however it won't be anything meaningful.

So, the next line (`neuralnet_initialize(...)`) will initialize some random weights such that
the values propagated through the net does neither vanish or explode. The function
`neuralnet_initialize()` takes in a pointer the neural network structure, followed by a
string array describing the initialization method to be used.
Calling `neuralnet_initalize( nn, NULL );` will pick reasonable initialization methods,
based on the activation function of each layer.

However, it is not able to do training with this neural network yet. 
The next line is then setting a loss function. A loss function is the function that will
be use for the optimization of the parameters of the neural network. This is therefore
necessary before we can do any training of the neural network. We have settled for
"binary_crossentropy" in this case, which is nice for classifications with sigmoid output
activations. (There is actually a mathematical reason for this good match. Try to find the
derivative of the loss function and the derivative of the output activation. If you do
that mathematical exercise, you will see the beauty.)

### Training the neural network.

So we are now ready to train the neural network and we will use an update rule called
Stochastic Gradient Descent. We will loop through the training set once. One iteration
of training through a training dataset is called an epoch, so we will train for one epochs.

    ...
        const int n_train_samples = train_X->shape[0];
        const int n_features      = train_X->shape[1];
        const int n_parameters    = neuralnet_total_n_parameters( nn );
        const float learning_rate = 0.01f;
    
        float SIMD_ALIGN(gradient[n_parameters]); 
    
        float *train_feature = (float*) train_X->data;
        float *train_target  = (float*) train_Y->data;
        for( int i = 0; i < n_train_samples; i++ ){
            neuralnet_backpropagation( nn, train_feature, train_target, gradient );
            for( int j = 0; j < n_parameters; j++ )
                gradient[j] *= -learning_rate;
            neuralnet_update( nn, gradient );
            train_feature += n_features;
            train_target  += 1;
        }
        printf("Done.\n");
    ...

Does that look overwhelming? Relax... the next paragraphs will simplify a bit. If you know
something about C as a programming language and something about Stochastic Gradient Descent,
then the above code lines should make somewhat sense.

### Evaluating the current neural network

How good is this neural network now? Let's loop through the **test** partition and predict
every mushroom sample with the neural network and then compare it with the given target.
If it predicts rights for most sample we are pretty sure it must have learned something.
Let's calculate the ratio of correctly classified mushrooms with the total number of
test samples. This will be what data scientists will call the "accuracy" or
"binary accuracy". A wild guess form the neural network will give about an accuracy
of 0.5 since it is about 50% chance of guessing right. We must hope we get a higher
accuracy then 0.5.

    ...
        int correct_count = 0;
        const int n_test_samples = test_X->shape[0];
        
        float *test_feature = (float*) test_X->data;
        float *test_target  = (float*) test_Y->data;
        for( int i = 0; i < n_test_samples; i++ ){
            float output[1];
            neuralnet_predict( nn, test_feature, output );
            int y_pred = output[0] > 0.5f ? 1 : 0;
            int y_true = *test_target > 0.5f ? 1 : 0;
            if( y_pred == y_true )
                correct_count++;
            test_feature += n_features;
            test_target  += 1;
        }
    
        printf("Accuracy: %5.5f\n", (float) correct_count / (float) n_test_samples );
    ...

### Clean up resources

We have created resources for the neural network itself and the training and test data.
The only thing left to do is to free these resources and return to the shell.

        ...
        neuralnet_free( nn );
        npy_array_list_free( filelist );
        return 0;
    }

### Compile and run
(Note to self: We need a configure/makefile here such that the -mavx parameter gets included.)

    gcc -std=c99 -Wall -Wextra -O3 -I../neuralnet -I../npy_array -c example_01.c
    gcc -o example_01 example_01.o -L../neuralnet -lneuralnet -L../npy_array -lnpy_array -lm

    ./example_01

Your mileage may vary, but I get accuracy of **0.99795**. Really not bad.

### Using optimizers and callbacks.
(in progress)
As you see in the code above, the train loop and the evaluation loop is really cumbersome and
inflexible. We have therefore developed a set of optimizers. Optimizers are are the routines
that gets the gradient of the loss wrt. the parameters and applies a rule to update the
parameters in the neural network. There are several standard optimizers the scientists have
developed over the years. Currently the following optimisers has been implemented in this
projects:

 * Stochastic Gradient Descent (SGD)
 * RMSprop
 * Adagrad
 * Adam
 * AdamW

Let's first make the manual implementation above use the supplied SGD code.

    ...
        /* Training with plain Stochastic Gradient Decsent (SGD) */
        const int n_train_samples = train_X->shape[0];
        const int n_test_samples = test_X->shape[0];
        const float learning_rate = 0.01f;
    
        optimizer_t *sgd = optimizer_new( nn,
                OPTIMIZER_CONFIG(
                    .batchsize = 1,
                    .shuffle   = false,
                    .run_epoch = SGD_run_epoch,
                    .settings  = SGD_SETTINGS( .learning_rate = learning_rate ),
                    .metrics   = ((metric_func[]){ get_metric_func( get_loss_name( nn->loss ) ),
                        get_metric_func( "binary_accuracy" ), NULL }),
                    .progress  = NULL
                    )
                );
    
        float *train_feature = (float*) train_X->data;
        float *train_target  = (float*) train_Y->data;
        float *test_feature  = (float*) test_X->data;
        float *test_target   = (float*) test_Y->data;
    
        int n_metrics = optimizer_get_n_metrics( sgd );
        float results[2*n_metrics];
    
        optimizer_run_epoch( sgd, n_train_samples, train_feature, train_target,
                                  n_test_samples,  test_feature,  test_target, results );
    
        printf("Train loss    : %5.5f\n", results[0] );
        printf("Train accuracy: %5.5f\n", results[1] );
        printf("Test loss     : %5.5f\n", results[2] );
        printf("Test accuracy : %5.5f\n", results[3] );
    ...

As you see there are some parameters to setup a optimizer,
but when it's done it basically does both loops for you.
It does the training loop (a training epoch) and it does the
evaluation. As you create the optimizer with `optimizer_new()`,
you also pass in the metrics that will be used in the evaluation
loop. In the above example, we've passed in the binary_crossentropy
metric and the binary accuracy. ´n_metrics´ will therfore be 2 in
this case. We will evaluate for both the train partition and the
test partition so the result parameter to the optimizer must have
space for four float values.

(Discuss)
## Other examples

(Work in progress)
