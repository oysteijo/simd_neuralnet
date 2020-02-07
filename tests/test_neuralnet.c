#include "test.h"
#include "neuralnet.h"
#include <stdio.h>

int main(int argc, char *argv[] )
{
    int test_count = 0;
    int fail_count = 0;

    neuralnet_t *nn = neuralnet_create( 2,
            INT_ARRAY( 3,4,2 ),
            STR_ARRAY( "tanh", "sigmoid" ));

    CHECK_NOT_NULL_MSG( nn,
            "Checking that neural network was created" );
    
    CHECK_INT_EQUALS_MSG( 26, neuralnet_total_n_parameters(nn),
            "Checking that total number of parametes are 26" );
            
    CHECK_INT_EQUALS_MSG( 2, neuralnet_get_n_layers(nn),
            "Checking that total number of layers are 2" );
            
    neuralnet_free( nn );

    /* Just testing some 'misconfigurations' */
    neuralnet_t *nn2 = neuralnet_create( 2,
            INT_ARRAY( 3,4,2, -1 ),
            STR_ARRAY( NULL, "sigmoid", "blah" )); /* Should work? */

    CHECK_NOT_NULL_MSG( nn2,
            "Checking that neural network was created" );
    
    CHECK_INT_EQUALS_MSG( 26, neuralnet_total_n_parameters(nn2),
            "Checking that total number of parametes are 26" );
            
    CHECK_INT_EQUALS_MSG( 2, neuralnet_get_n_layers(nn2),
            "Checking that total number of layers are 2" );
            
    neuralnet_free( nn2 );

    /* Test initialization */
    /* Make a big neural network such that the central limit theorem can
     * help us evaluate the correctness. */
    nn = neuralnet_create( 2,
            INT_ARRAY( 1000, 1000, 10 ),
            STR_ARRAY( NULL ) );

    CHECK_NOT_NULL_MSG( nn,
            "Checking that neural network was created" );

    neuralnet_initialize( nn, "kaiming", "kaiming");  /* Normal distribution - scale sqrtf(2.0f/n_inp) */

    float *params = NULL;
    neuralnet_get_params( nn, params );

    neuralnet_free( nn );

    printf("Total test done  : %d\n", test_count );
    printf("Total test failed: %d\n", fail_count );
    return 0;
}
    
