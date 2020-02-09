#include "test.h"
#include "neuralnet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char *argv[] )
{
    int test_count = 0;
    int fail_count = 0;

    fprintf(stderr, KBLU "Testing neuralnet_create." KNRM "\n" );
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
    fprintf(stderr, KBLU "Testing initialisation with a bigger neuralnet." KNRM "\n" );
    /* Make a big neural network such that the central limit theorem can
     * help us evaluate the correctness. */
    const int n_inp = 1000;
    const int n_out = 1000;
    nn = neuralnet_create( 2,
            INT_ARRAY( n_inp, n_out, 10 ),
            STR_ARRAY( "tanh", "sigmoid" ) );

    CHECK_NOT_NULL_MSG( nn,
            "Checking that neural network was created" );

    srand(time(0));

    /* Check Kaiming initializing. */
    fprintf(stderr, KBLU "Testing Kaiming initialisation" KNRM "\n" );
    neuralnet_initialize( nn, "kaiming", "kaiming");  /* Normal distribution - scale sqrtf(2.0f/n_inp) */

    int n_params = neuralnet_total_n_parameters(nn);

    CHECK_INT_EQUALS_MSG( (n_inp*n_out)+(n_out*10)+n_out+10, n_params,
           "Checking number of parameters in a bigger neural network" );

    float *params = malloc( n_params * sizeof(float));
    if( !params ){
        fprintf(stderr, "Cannot allocate memory for testing 'parameters'. (Aborting test)\n" );
        goto end_of_tests;
    }

    neuralnet_get_parameters( nn, params );
    float mean = test_calculate_mean( n_inp*n_out, params+n_out );  /* Adding n_out since the first thousand is bias */
    float stddev = test_calculate_stddev( n_inp*n_out, params+n_out );  /* Adding n_out since the first thousand is bias */


    CHECK_FLOAT_EQUALS_MSG( mean, 0.0f, 1.0e-4f,
            "Checking that mean of parameters are actually 0.0" ); 
    fprintf(stderr, "\tmean of parameters   : %+g\n", mean);

    CHECK_FLOAT_EQUALS_MSG( stddev, sqrtf( 2.0f / n_inp ), 1.0e-4f,
            "Checking that stddev of parameters are actually sqrt( 2/n_inp )" );
    fprintf(stderr, "\tstddev of parameters : %+g\n", stddev);

#if 0
    fprintf(stderr, "The max and min values sould be about +0.2 and - 0.2 for the 1000x1000 case.\n");
    fprintf(stderr, "\tmaximum of parameters: %+g\n", test_calculate_max( n_inp*n_out, params+n_out ));
    fprintf(stderr, "\tminimum of parameters: %+g\n", test_calculate_min( n_inp*n_out, params+n_out ));
#endif

    /* Check Xavier initializing. */
    fprintf(stderr, KBLU "Testing Xavier initialisation" KNRM "\n" );

    /* Xavier initialization us uniform within a and b where a = sqrt(6/(n_inp+n_out)). The tests we simply do
     * is to check the mean, stddev and that the max(x) and min(x) are inside the interval. */
    neuralnet_initialize( nn, "xavier", "xavier");  /* Uniform distribution - scale sqrtf(6.0f/(n_inp+n_out)) */
    neuralnet_get_parameters( nn, params );

    mean = test_calculate_mean( n_inp*n_out, params+n_out );  /* Adding n_out since the first thousand is bias */
    stddev = test_calculate_stddev( n_inp*n_out, params+n_out );  /* Adding n_out since the first thousand is bias */

    CHECK_FLOAT_EQUALS_MSG( mean, 0.0f, 1.0e-4f,
            "Checking that mean of parameters are actually 0.0" ); 
    fprintf(stderr, "\tmean of parameters   : %+g\n", mean);

    float a = -sqrtf( 6.0 /(n_inp+n_out));
    float b = -a; /* sqrtf( 6.0 /(n_inp+n_out)); */

    float stddev_theoretical = (b-a) / sqrtf(12);

    CHECK_FLOAT_EQUALS_MSG( stddev, stddev_theoretical, 1.0e-4f,
            "Checking that stddev of parameters are actually (b-a)/sqrt(12)" );
    fprintf(stderr, "\tstddev of parameters : %+g\n", stddev);

    float minval = test_calculate_min( n_inp*n_out, params+n_out );
    float maxval = test_calculate_max( n_inp*n_out, params+n_out );

    CHECK_CONDITION_MSG( minval >= a, 
            "Checking that min value is higher than theoretical minimum" );

    CHECK_CONDITION_MSG( maxval <= b, 
            "Checking that max value is higher than theoretical maximum" );

    CHECK_FLOAT_EQUALS_MSG( minval, a, 1.0e-04f,  /* Just an abritary value. */
            "Checking that the minimum value is not far off the theoretical");

    CHECK_FLOAT_EQUALS_MSG( maxval, b, 1.0e-04f,
            "Checking that the maximum value is not far off the theoretical");

    fprintf(stderr, "\tmaximum of parameters: %+g\n", maxval );
    fprintf(stderr, "\ttheoretical maximum  : %+g\n", b );
    fprintf(stderr, "\tminimum of parameters: %+g\n", minval );
    fprintf(stderr, "\ttheoretical minimum  : %+g\n", a );
    free( params );

    /* TODO Still left to test load nad save */

end_of_tests:
    neuralnet_free( nn );

    print_test_summary(test_count, fail_count );
    return 0;
}
