#include "adamw.h"
#include "c_npy.h"
#include "neuralnet.h"
#include "activation.h"
#include "loss.h"
#include "metrics.h"

#include "strtools.h"
#include "progress.h"

/* Callbacks */
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
    assert( test_X->fortran_order  == false );
    assert( test_Y->fortran_order  == false );

    const int n_train_samples = train_X->shape[0];
    const int n_test_samples = test_X->shape[0];

    /* It does not handle any whitespace in front of (or trailing) */
    char **split = strsplit( argv[2], ',' );
    neuralnet_t *nn = neuralnet_new( argv[1], split );
    strsplit_free(split);
    assert( nn );
    neuralnet_set_loss( nn, "binary_crossentropy" );


    optimizer_t *adamw = optimizer_new( nn, 
            OPTIMIZER_CONFIG(
                .batchsize = 16384,
                .shuffle   = true,
                .run_epoch = adamw_run_epoch,
                .settings  = ADAMW_SETTINGS( .learning_rate = 0.001f, .weight_decay=1e-5f ),
                .metrics   = ((metric_func[]){ get_metric_func( "mean_squared_error"), NULL })
                )
            );

    int n_metrics = optimizer_get_n_metrics( adamw );

    int n_epochs = 100;
    float *results = calloc( 2 * n_metrics * n_epochs, sizeof(float)); /* I usually won't need one each epoch */

    /* Make some callbacks */
    callback_t *logger     = CALLBACK(logger_new         ( LOGGER_NEW         ( .filename="adamw.log" ) ));
    callback_t *checkpoint = CALLBACK(modelcheckpoint_new( MODELCHECKPOINT_NEW( ) ));
    callback_t *earlystop  = CALLBACK(earlystopping_new  ( EARLYSTOPPING_NEW  ( .patience = 15 ) ));

    /* Set the callbacks in an array */
    callback_t *cbarray[] = { logger, checkpoint, earlystop };
    const int n_callbacks = sizeof( cbarray ) / sizeof( cbarray[0] );

    for ( int i = 0; i < n_epochs && !earlystopping_do_stop( EARLYSTOPPING(earlystop)) ; i++ ){
        /* Run the optimizer for one epoch */
        optimizer_run_epoch( adamw, n_train_samples, (float*) train_X->data, (float*) train_Y->data,
                                  n_test_samples,  (float*) test_X->data, (float*) test_Y->data, results+2*i );

        /* Run all callbacks */
        for( int j = 0; j < n_callbacks; j++ )
            callback_run( cbarray[j], adamw, results+2*i, true );
    }

    /* Cleanup */
    for( int j = 0; j < n_callbacks; j++ )
        callback_free( cbarray[j] );

    free(results);
    neuralnet_free( nn );
    optimizer_free( adamw );
    c_npy_matrix_array_free( train_test );
    return 0;
}    
    
