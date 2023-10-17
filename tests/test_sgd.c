#include "neuralnet.h"
#include "optimizer.h"
#include "optimizer_implementations.h"
#include "simd.h"
#include "loss.h"
#include "metrics.h"
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
            INT_ARRAY( 64, 32, 16, 8, 4 ),   /* The 8 will be resized on avx512 */
            STR_ARRAY( "relu", "relu", "relu", "sigmoid" )
    );

    assert( nn );

    neuralnet_initialize( nn, NULL );
    neuralnet_set_loss( nn, "binary_crossentropy" );

    unsigned int n_params = neuralnet_total_n_parameters( nn );
    float SIMD_ALIGN(grad[n_params]);
    float SIMD_ALIGN(pre_update_params[n_params]);
    float SIMD_ALIGN(post_update_params[n_params]);

    /* I don't think these needs to be aligned, however we do it anyway! */
    float SIMD_ALIGN(input[64]);
    float SIMD_ALIGN(target[4]);

    for( int i = 0; i < 64; i++)
        input[i] = 1.0f;

    for( int i = 0; i < 4; i++)
        target[i] = 1.0f;

    float learning_rate = 0.01f;
    optimizer_t *sgd = optimizer_new( nn, 
            OPTIMIZER_CONFIG(
                .batchsize = 16,
                .shuffle   = false,
                .run_epoch = SGD_run_epoch,
                .settings  = SGD_SETTINGS( .learning_rate = learning_rate ),
                .metrics   = ((metric_func[]){ get_metric_func( get_loss_name( nn->loss ) ),
                    get_metric_func( "mean_squared_error"),
                    get_metric_func( "binary_accuracy" ), NULL }),
                .progress  = NULL
                )
            );

    int n_metrics = optimizer_get_n_metrics( sgd );

    neuralnet_backpropagation( nn, input, target, grad );
    neuralnet_get_parameters(  nn, pre_update_params );

    /*
    void optimizer_run_epoch( optimizer_t *self,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y,
        const unsigned int n_valid_samples, const float *valid_X, const float *valid_Y, float *result );
    */

    float result[2*n_metrics];
    optimizer_run_epoch( sgd, 1, input, target, 0, NULL, NULL, result );
    neuralnet_get_parameters( nn, post_update_params );

    int n_fail = 0;
    int n_tests = 0;
    int print_col = 0;
    const float epsilon = 1.0e-06f;  /* We do indeed expect some deviation. */
    for ( unsigned int i = 0; i < n_params; i++ ){
        float delta = (post_update_params[i] - pre_update_params[i]);
        if( fabsf( delta + learning_rate*grad[i] ) > epsilon ){
            fprintf( stderr, KRED "\ndelta_w != -learningi_rate * grad   (%5.5f != %5.5f)\n" KNRM, delta, -learning_rate*grad[i] );
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

    fprintf(stderr, "Stochastic Gradient Descent test report, using epsilon=%g and learning rate=%g\n", epsilon, learning_rate);
    fprintf(stderr, "%5d tests done\n", n_tests );
    fprintf(stderr, "%s%5d tests failed\n" KNRM, n_fail ? KRED : KGRN, n_fail );

    return 0;
}

