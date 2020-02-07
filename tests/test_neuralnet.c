#include "test.h"
#include "neuralnet.h"
#include <stdlib.h>
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

    int n_params = neuralnet_total_n_parameters(nn);

    CHECK_INT_EQUALS_MSG( 1000000+10000+1000+10, n_params,
           "Checking number of parameters in a bigger neural network" );

    float *params = malloc( n_params * sizeof(float));
    if( !params ){
        fprintf(stderr, "Cannot allocate memory for testing 'parameters'. (Aborting test)\n" );
        goto end_of_tests;
    }

    neuralnet_get_parameters( nn, params );
    float mean = test_calculate_mean( 1000*1000, params+1000 );  /* Adding 1000 since the first thousand is bias */
    float stddev = test_calculate_stddev( 1000*1000, params+1000 );  /* Adding 1000 since the first thousand is bias */


    CHECK_FLOAT_EQUALS_MSG( mean, 0.0f, 1.0e-5f,
            "Checking that mean of parameters are actually 0.0" ); 
    fprintf(stderr, "\tmean of parameters: %g\n", mean);
    fprintf(stderr, "\tstddev of parameters: %g\n", stddev);

    fprintf(stderr, "\tmaximum of parameters: %g\n", test_calculate_max( 1000*1000, params+1000 ));
    fprintf(stderr, "\tminimum of parameters: %g\n", test_calculate_min( 1000*1000, params+1000 ));

    free( params );

end_of_tests:
    neuralnet_free( nn );

    printf("Total test done  : %d\n", test_count );
    printf("Total test failed: %d\n", fail_count );
    return 0;
}
