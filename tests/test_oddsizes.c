#include "test.h"
#include "neuralnet.h"
#include "simd.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

struct nntest {
    int  *sizes;
    char **activations;
    char *loss;
} test_sizes[3] = {
    { .sizes = INT_ARRAY( 231, 128, 5),
      .activations = STR_ARRAY("relu", "sigmoid"),
      .loss = "binary_crossentropy" },
    { .sizes = INT_ARRAY( 53, 19, 13, 7),
      .activations = STR_ARRAY("relu", "tanh", "sigmoid"),
      .loss = "binary_crossentropy"},
    { .sizes = INT_ARRAY( 103, 53, 19, 13, 7),
      .activations = STR_ARRAY("relu", "hard_sigmoid", "tanh", "softmax"),
      .loss = "categorical_crossentropy" },
};

int main(int argc, char *argv[] )
{
    int test_count = 0;
    int fail_count = 0;

    if(argc == 1)
        fprintf(stderr, KBLU "Running '%s'\n" KNRM, argv[0] );

    for( int i = 0; i < 3; i++ ){
        fprintf(stderr, KBLU "Creating neural network %d\n" KNRM, i+1 );
        neuralnet_t *nn = neuralnet_create( i+2,
                test_sizes[i].sizes, test_sizes[i].activations);
    
        CHECK_NOT_NULL_MSG( nn,
                "Checking that neural network was created" );

        int nn_n_input = nn->layer[0].n_input;
        int nn_n_output = nn->layer[nn->n_layers-1].n_output;

        neuralnet_initialize(nn, NULL);

        float *input = simd_malloc( nn_n_input * sizeof(float) );
        float *output = simd_malloc( nn_n_output * sizeof(float) );
        float *target = simd_malloc( nn_n_output * sizeof(float) );
        float *grad = simd_malloc( neuralnet_total_n_parameters(nn) * sizeof(float) );

        /* Try forward calc. */
        for( int j = 0; j < nn_n_input; j++)
            input[j] = 1.0f;

        neuralnet_predict( nn, input, output);
        printf("Predictions:\n");
        for( int j = 0; j < nn_n_output; j++)
            printf("%6.6f\n", output[j]);
        
        /* Try one backward calc. */
        neuralnet_set_loss( nn, test_sizes[i].loss );
        for( int j = 0; j < nn_n_output; j++)
            target[j] = 1.0f;
        neuralnet_backpropagation( nn, input, target, grad );

        printf("Some random gradient value: %6.6f\n", grad[10]);
        /*  clean up */
        simd_free( input );
        simd_free( output );
        simd_free( target );
        simd_free( grad );

        neuralnet_free( nn );   
    }
    print_test_summary(test_count, fail_count );
    return 0;
}
