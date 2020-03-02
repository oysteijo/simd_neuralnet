#include "npy_array.h"
#include "neuralnet.h"
#include "simd.h"

#include "optimizer.h"
#include "optimizer_implementations.h"
#include "loss.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main( int argc, char *argv[] )
{
    /* Read the datafile created in python+numpy */
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

    /* Set up a new Neural Network */
    neuralnet_t *nn = neuralnet_create( 3,
            INT_ARRAY( train_X->shape[1], 64, 32, 1 ),
            STR_ARRAY( "relu", "relu", "sigmoid" ) );
    assert( nn );

    neuralnet_initialize( nn, STR_ARRAY("kaiming", "kaiming", "kaiming"));
    neuralnet_set_loss( nn, "binary_crossentropy" );


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

    float results[ 2 * optimizer_get_n_metrics( sgd ) ];
    optimizer_run_epoch( sgd, n_train_samples, (float*) train_X->data, (float*) train_Y->data,
                              n_test_samples,  (float*) test_X->data, (float*) test_Y->data, results );

    printf("Train loss    : %5.5f\n", results[0] );
    printf("Train accuracy: %5.5f\n", results[1] );
    printf("Test loss     : %5.5f\n", results[2] );
    printf("Test accuracy : %5.5f\n", results[3] );

    /* Clean up the resources */
    neuralnet_free( nn );
    npy_array_list_free( filelist );
    return 0;
}

