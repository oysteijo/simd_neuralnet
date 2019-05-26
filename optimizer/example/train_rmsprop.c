#include "RMSprop.h"
#include "c_npy.h"
#include "neuralnet.h"
#include "activation.h"
#include "loss.h"
#include "metrics.h"

#include "strtools.h"
#include "progress.h"

#include "logger.h"
#include "modelcheckpoint.h"
#include "earlystopping.h"

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
    cmatrix_t *test_X  = train_test[2];
    cmatrix_t *test_Y  = train_test[3];

    /* if any of these asserts fails, try to open the weight in python and save with
     * np.ascontiguousarray( matrix ) */
    assert( train_X->fortran_order == false );
    assert( train_Y->fortran_order == false );
    assert( test_X->fortran_order == false );
    assert( test_Y->fortran_order == false );

    const int n_train_samples = train_X->shape[0];
    const int n_test_samples = test_X->shape[0];

    /* It does not handle any whitespace in front of (or trailing) */
    char **split = strsplit( argv[2], ',' );
    neuralnet_t *nn = neuralnet_new( argv[1], split );
    strsplit_free(split);
    assert( nn );
    neuralnet_set_loss( nn, "binary_crossentropy" );


    optimizer_t *RMSprop = optimizer_new( nn, 
            OPTIMIZER_CONFIG(
                .batchsize = 256,
                .shuffle   = true,
                .run_epoch = RMSprop_run_epoch,
                .settings  = RMSPROP_SETTINGS( .learning_rate = 0.01f ),
                .metrics   = ((metric_func[]){ get_metric_func( "mean_squared_error"), NULL })
                )
            );

    int n_metrics = optimizer_get_n_metrics( RMSprop );

    int n_epochs = 10;
    srand( 70 );
    float *results = calloc( 2 * n_metrics * n_epochs, sizeof(float));

    logdata_t logdata = {
        .epoch_count = 1,
        .no_stdout   = false, 
        .filename    = NULL 
    };

    checkpointdata_t cpdata = {
        .filename = NULL,
        .greater_is_better = false,
        .monitor_idx = -1,
        .verbose  = false

    };
    
    earlystoppingdata_t esdata = {
        .patience          = 10,
        .greater_is_better = false,
        .monitor_idx       = -1,
        .early_stopping_flag = false
    };

    
    for ( int i = 0; i < n_epochs && !esdata.early_stopping_flag; i++ ){
        optimizer_run_epoch( RMSprop, n_train_samples, (float*) train_X->data, (float*) train_Y->data,
                                  n_test_samples,  (float*) test_X->data, (float*) test_Y->data, results+2*i );

        logger( RMSprop, results+2*i, true, (void *) &logdata );
        // modelcheckpoint( RMSprop, results+2*i, true, (void *) &cpdata );
        // earlystopping( RMSprop, results+2*i, true, (void*) &esdata );
        // printf("epoch: %d  mse: %e  val_mse: %e\n", i, results[2*i], results[2*i+1]);
    }

    free(results);
    /* log and report */
    neuralnet_save( nn, "after-10-epochs.npz" );
    neuralnet_free( nn );
    free( RMSprop );
    c_npy_matrix_array_free( train_test );
    return 0;
}    
    
