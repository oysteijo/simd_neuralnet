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
    int n_input = 214;
    int n_output = 5;

#if 1
    neuralnet_t *nn = neuralnet_create( 2,
            INT_ARRAY( n_input, 64, n_output ),
            STR_ARRAY( "tanh", "sigmoid" ) );

    assert( nn );
    neuralnet_initialize( nn, NULL);
    neuralnet_set_loss( nn, "mean_squared_error" );
#else
    neuralnet_t *nn = neuralnet_create( 4,
            INT_ARRAY( 64, 32, 16, 8, 4 ),   /* The 8 will be resized on a AVX512 machine */
            STR_ARRAY( "relu", "relu", "relu", "sigmoid" )
    );

    assert( nn );
    neuralnet_initialize( nn, NULL );
    neuralnet_set_loss( nn, "binary_crossentropy" ); 
#endif
    metric_func loss = get_metric_func( get_loss_name( nn->loss ) );  /* neuralnet_get_loss() ? */ 

    unsigned int n_params = neuralnet_total_n_parameters( nn );
    float SIMD_ALIGN(grad0[n_params]);
    float SIMD_ALIGN(grad1[n_params]);
    float SIMD_ALIGN(delta_param[n_params]);
    memset( delta_param, 0, n_params * sizeof(float ));

    float SIMD_ALIGN(input[n_input]);
    float SIMD_ALIGN(target[n_output]);
    float SIMD_ALIGN(output[n_output]);

    assert( is_aligned( input ));
    assert( is_aligned( target ));
    assert( is_aligned( output ));

    /* an example input vector */
    for( int i = 0; i < n_input; i++)
        input[i] = 1.0f;

    /* an example target vector */
    for( int i = 0; i < n_output; i++)
        target[i] = 1.0f;

    printf("addr of inp: %p  (aligned: %s)\n", input, is_aligned( input ) ? "True" : "False");
    neuralnet_predict( nn, input, output );
    float J0 = loss(n_output, output, target );

    neuralnet_backpropagation( nn, input, target, grad0 );

    int n_fail = 0;
    int n_tests = 0;
    int print_col = 0;
    const float delta   = 1.0e-04f;  /* If this is set too small, J1 - J0 becomes too small maybe even 0... */
    const float epsilon = 1.0e-02f;  /* We don't expect the numerical to match the backprop grad. better. */
    for ( unsigned int i = 0; i < n_params; i++ ){
        delta_param[i] = delta;
        neuralnet_update( nn, delta_param );
        neuralnet_predict( nn, input, output );
        neuralnet_backpropagation( nn, input, target, grad1 );

        delta_param[i] = -delta;
        neuralnet_update( nn, delta_param );
        delta_param[i] = 0.0f;

        float grad = 0.5f * (grad0[i] + grad1[i]);

        float J1 = loss(n_output, output, target );
        // printf("J0: %5.6f  J1: %5.6f\n", J0, J1 );
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
        // break;
    }
    fprintf(stderr, "\n" );

    fprintf(stderr, "Gradient test report, using delta=%g and epsilon=%g\n", delta, epsilon);
    fprintf(stderr, "%5d tests done\n", n_tests );
    fprintf(stderr, "%s%5d tests failed\n" KNRM, n_fail ? KRED : KGRN, n_fail );

	fprintf(stderr, "Please note - this test of the gradient calculation is really inaccurate. There can\n"
			"be significant discrepansies between the gradient values from the backpropagation\n"
			"algorithm and the numerical calculation.\n\n"
			"If you have Keras installed and you have any doubt about the calculation of the gradient,\n"
			"you should rather run 'test_keras.py'. That python code will set up a Keras neural network,\n"
			"and select a random sample and a random taget vector and calculate the gradient using Keras.\n"
			"The model, the sample/target and the gradient are then saved in .npz files. You can then test\n"
			"this gradient with the `test_backgammon_files` executable. That's a better check!");

    return 0;
}

