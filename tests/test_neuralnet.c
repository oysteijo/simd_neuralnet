#include "test.h"
#include "neuralnet.h"
#include "simd.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

int main(int argc, char *argv[] )
{
    int test_count = 0;
    int fail_count = 0;

    if(argc == 1)
        fprintf(stderr, KBLU "Running '%s'\n" KNRM, argv[0] );

    fprintf(stderr, KBLU "Testing neuralnet_create." KNRM "\n" );
    neuralnet_t *nn = neuralnet_create( 2,
            INT_ARRAY( 3,4,2 ),
            STR_ARRAY( "tanh", "sigmoid" ));

    CHECK_NOT_NULL_MSG( nn,
            "Checking that neural network was created" );
    
    /* The neural network might have been resized to align memory better.
       In that case the number of parameters is not correct. I might have to
       find a better test that takes into account the ALIGN_SIZE */
    fprintf(stderr, "This instruction set can do " KGRN "%d" KNRM " floats pr. SIMD register",
            floats_per_simd_register);
    fprintf(stderr, " - %s maybe.\n",
            floats_per_simd_register==16 ? "AVX512" : 
            floats_per_simd_register==8 ? "AVX/AVX2" : 
            floats_per_simd_register==4 ? "SSE" : "No SIMD");
    const int n_param_342 = 26;
    printf( "I believe the number of parameters should be: %d\n", n_param_342 );

    CHECK_INT_EQUALS_MSG( n_param_342, neuralnet_total_n_parameters(nn),
            "Checking that total number of parametes is correct" );
            
    CHECK_INT_EQUALS_MSG( 2, neuralnet_get_n_layers(nn),
            "Checking that total number of layers are 2" );
            
    neuralnet_free( nn );

    /* Just testing some 'misconfigurations' */
    neuralnet_t *nn2 = neuralnet_create( 2,
            INT_ARRAY( 3,4,2, -1 ),
            STR_ARRAY( NULL, "sigmoid", "blah" )); /* Should work? */

    CHECK_NOT_NULL_MSG( nn2,
            "Checking that neural network was created" );
    
    CHECK_INT_EQUALS_MSG( n_param_342, neuralnet_total_n_parameters(nn2),
            "Checking that total number of parametes is correct" );
            
    CHECK_INT_EQUALS_MSG( 2, neuralnet_get_n_layers(nn2),
            "Checking that total number of layers are 2" );
            
    neuralnet_free( nn2 );

    /* Test initialization */
    fprintf(stderr, KBLU "Testing initialisation with a bigger neuralnet." KNRM "\n" );
    /* Make a big neural network such that the central limit theorem can
     * help us evaluate the correctness. */
    int n_inp = 1000;
    int n_out = 1000;
    nn = neuralnet_create( 2,
            INT_ARRAY( n_inp, n_out, 10 ),
            STR_ARRAY( "tanh", "sigmoid" ) );

    CHECK_NOT_NULL_MSG( nn,
            "Checking that neural network was created" );

    srand(time(0));

    /* Check Kaiming initializing. */
    fprintf(stderr, KBLU "Testing Kaiming initialisation" KNRM "\n" );
    neuralnet_initialize( nn, STR_ARRAY("kaiming", "kaiming") );  /* Normal distribution - scale sqrtf(2.0f/n_inp) */

    int n_params = neuralnet_total_n_parameters(nn);

    // printf("%d ?= %d \n", (n_inp*n_out)+(n_out*10)+n_out+10, n_params);
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
    neuralnet_initialize( nn, STR_ARRAY( "xavier", "xavier" ));  /* Uniform distribution - scale sqrtf(6.0f/(n_inp+n_out)) */
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

    /* Just for testing... let's try to re-initialize with a null-pointer. That should work now! */
    neuralnet_initialize( nn, NULL ); /* Should not crash! */
    CHECK_NOT_NULL_MSG( nn,
            "Checking that neural network can be re-initialized" );
    /* Wow! That didn't work the first time! Good we tested! */

    /* predict, save, load and predict again */
    fprintf(stderr, KBLU "Testing prediction, saving, loading and then predicting again." KNRM "\n" );
    float *inp = simd_malloc( n_inp * sizeof(float));
    assert(inp);
    for( int i = 0; i < n_inp; i++ )
        inp[i] = 0.5f;

    float SIMD_ALIGN(result[10]);
    
    fprintf(stderr, "Predicting 10 values:\n" );
    neuralnet_predict( nn, inp, result );
    for( int i = 0; i < 10; i++ )
        fprintf(stderr, "%5.5f ", result[i] );
    fprintf(stderr, "\n" );

    fprintf(stderr, "Saving neural network.\n" );

    neuralnet_save( nn, "tmp_store_%d.npz", 12 );
    CHECK_CONDITION_MSG( access( "tmp_store_12.npz", F_OK ) != -1,
            "Checking that file got saved" );
    /* free the neural network and reopen. Does it still predict the same values? */
    neuralnet_free( nn );

    nn = neuralnet_load( "tmp_store_12.npz" );
    CHECK_NOT_NULL_MSG( nn,
            "Checking that neural network was created from load" );
    float SIMD_ALIGN(result_new[10]);

    fprintf(stderr, "Predicting same 10 values from same inputs:\n" );
    neuralnet_predict( nn, inp, result_new );
    for( int i = 0; i < 10; i++ ){
        char buffer[256];
        sprintf(buffer, "%3d : %5.5f ?= %5.5f", i, result[i], result_new[i] );
        CHECK_FLOAT_EQUALS_MSG( result[i], result_new[i], 1.0e-07, buffer );
    }

    fprintf(stderr, "\n" );

    free( inp );

end_of_tests:
    neuralnet_free( nn );

    print_test_summary(test_count, fail_count );
    return 0;
}
