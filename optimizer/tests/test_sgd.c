#include "stochastic_gradient_descent.h"
#include "c_npy.h"
#include "neuralnet.h"
#include "activation.h"
#include "loss.h"
#include "metrics.h"

#include "strtools.h"
#include "progress.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

STRSPLIT_INIT
// STRSPLIT_LENGTH_INIT
STRSPLIT_FREE_INIT

int main( int argc, char *argv[] )
{
    if (argc != 4 ){
        fprintf( stderr, "Usage: %s <weightsfile.npz> <list of activations> <trainsamples.npz>\n", argv[0] );
        return 0;
    }

    cmatrix_t **train_test = c_npy_matrix_array_read( argv[3]);
    if( !train_test ) return -1;

    cmatrix_t *train_X = train_test[0];
    cmatrix_t *train_Y = train_test[1];
    cmatrix_t *test_X = train_test[2];
    cmatrix_t *test_Y = train_test[3];

    /* if any of these asserts fails, try to open the weight in python and save with
     * np.ascontiguousarray( matrix ) */
    assert( train_X->fortran_order == false );
    assert( train_Y->fortran_order == false );
    assert( test_X->fortran_order == false );
    assert( test_Y->fortran_order == false );

    const int n_train_samples = train_X->shape[0];
    const int n_test_samples = test_X->shape[0];


    printf( "train_X shape: (%zu, %zu)\n", train_X->shape[0], train_X->shape[1] );
    printf( "train_Y shape: (%zu, %zu)\n", train_Y->shape[0], train_Y->shape[1] );
    printf( "test_X shape: (%zu, %zu)\n", test_X->shape[0], test_X->shape[1] );
    printf( "test_Y shape: (%zu, %zu)\n", test_Y->shape[0], test_Y->shape[1] );
    /* It does not handle any whitespace in front of (or trailing) */
    char **split = strsplit( argv[2], ',' );
    neuralnet_t *nn = neuralnet_new( argv[1], split );
    strsplit_free(split);
    assert( nn );
    neuralnet_set_loss( nn, "binary_crossentropy" );


    optimizer_t *sgd = optimizer_new( nn, 
            OPTIMIZER_CONFIG(
                .batchsize = 8,
                .shuffle   = true,
                .run_epoch = SGD_run_epoch,
                .settings  = SGD_SETTINGS( ), // .learning_rate = 0.01f, .momentum = 0.9f, .nesterov = true ),
                .metrics   = ((metric_func[]){ get_metric_func( get_loss_name( nn->loss ) ),
                    get_metric_func( "mean_squared_error"),
                    get_metric_func( "binary_accuracy" ), NULL }),
                .progress  = NULL
                )
            );

    int n_metrics = optimizer_get_n_metrics( sgd );

    int n_epochs = 100;
    
    float results[2*n_metrics];
    
    for ( int i = 0; i < n_epochs; i++ ){
        optimizer_run_epoch( sgd, n_train_samples, (float*) train_X->data, (float*) train_Y->data,
                                  n_test_samples, (float*) test_X->data, (float*) test_Y->data, results );
        printf( " %3d", i);
        for ( int j = 0; j < 2*n_metrics ; j++ )
            printf("\t%7e", results[j] );
        printf("\n");
    }

    /* log and report */
    neuralnet_save( nn, "after-%d-epochs.npz", n_epochs );
    neuralnet_free( nn );
    free( sgd );
    c_npy_matrix_array_free( train_test );
    return 0;
}    
    
