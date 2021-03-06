#include "neuralnet.h"
#include "simd.h"
#include "metrics.h"
#include "loss.h"
#include "test.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

int main( int argc, char *argv[]) 
{
    /* This testing program could actually read a nn from a file.
     * Say you read command line inputs like:
     *
     *   ./test_backpropagation <file_with_nn.npz> <file_with_input.npy> <file_with_target.npy> --output=outputfile.txt
     *
     * BTW: all testfiles should actually take an option to redirect output.
     */

    neuralnet_t *nn = neuralnet_create( 4,
            INT_ARRAY( 64, 32, 16, 8, 4 ),
            STR_ARRAY( "relu", "relu", "relu", "sigmoid" )
    );

    assert( nn );

    neuralnet_initialize( nn, NULL );
    neuralnet_set_loss( nn, "binary_crossentropy" );
    metric_func loss = get_metric_func( get_loss_name( nn->loss ) );  /* neuralnet_get_loss() ? */ 

    unsigned int n_params = neuralnet_total_n_parameters( nn );
    float SIMD_ALIGN(grad0[n_params]);
    float SIMD_ALIGN(grad1[n_params]);
    float SIMD_ALIGN(delta_param[n_params]);
    memset( delta_param, 0, n_params * sizeof(float ));

    float SIMD_ALIGN(input[64]);
    float SIMD_ALIGN(target[4]);
    float SIMD_ALIGN(output[4]);

    for( int i = 0; i < 64; i++)
        input[i] = 1.0f;

    for( int i = 0; i < 4; i++)
        target[i] = 1.0f;

    neuralnet_predict( nn, input, output );
    float J0 = loss(4, output, target );
    neuralnet_backpropagation( nn, input, target, grad0 );

    int n_fail = 0;
    int n_tests = 0;
    int print_col = 0;
    const float delta   = 1.0e-03f;  /* If this is set too small, J1 - J0 becomes too small maybe even 0... */
    const float epsilon = 1.0e-03f;  /* We don't expect the numerical to match the backprop grad. better. */
    for ( unsigned int i = 0; i < n_params; i++ ){
        delta_param[i] = delta;
        neuralnet_update( nn, delta_param );
        neuralnet_predict( nn, input, output );
        neuralnet_backpropagation( nn, input, target, grad1 );

        delta_param[i] = -delta;
        neuralnet_update( nn, delta_param );
        delta_param[i] = 0.0f;
        

        float grad = 0.5f * (grad0[i] + grad1[i]);

        float J1 = loss(4, output, target );
        float grad_by_numeric = (J1 - J0) / delta;
        if( fabsf( grad - grad_by_numeric ) > epsilon ){
            fprintf( stderr, KRED "\ngrad0 != numeric value   (%5.5f != %5.5f)\n" KNRM, grad, grad_by_numeric );
            n_fail++;
            print_col = 0;
        } else {
            fprintf( stderr, KGRN "." KNRM );
            print_col++;
        }
        if( ((print_col+1) % 64 == 0) ){
            fprintf( stderr, "\n" );
            print_col = 0;
        }
        n_tests++;
    }
    fprintf(stderr, "\n" );

    fprintf(stderr, "Gradient test report, using delta=%g and epsilon=%g\n", delta, epsilon);
    fprintf(stderr, "%5d tests done\n", n_tests );
    fprintf(stderr, "%s%5d tests failed\n" KNRM, n_fail ? KRED : KGRN, n_fail );

    return 0;
}

