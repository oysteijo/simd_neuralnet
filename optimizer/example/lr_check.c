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

    /* It does not handle any whitespace in front of (or trailing) */
    char **split = strsplit( argv[2], ',' );
    neuralnet_t *nn = neuralnet_new( argv[1], split );
    assert( nn );
    neuralnet_set_loss( nn, "binary_crossentropy" );


    optimizer_t *sgd = optimizer_new( nn, 
            OPTIMIZER_CONFIG(
                .batchsize = 1,
                .shuffle = true,
                .run_epoch = SGD_run_epoch,
                .settings  = SGD_SETTINGS( .learning_rate = 0.002 ),
                .metrics   = ((metric_func[]){ get_metric_func( "mean_squared_error"), NULL })
                )
            );

    int n_metrics = optimizer_get_n_metrics( sgd );

    float lrates[20] = {0.001f, 0.005f, 0.01f, 0.02f, 0.03f, 0.05f, 0.06f, .07f, 0.08f, 0.09f, 0.1f, 0.15f, 0.2, 0.25f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f }; 
    int n_trials = 20;
    srand( 70 );
    float *results = calloc( n_metrics * n_trials, sizeof(float));
    
    for ( int i = 0; i < n_trials; i++ ){

        sgd_settings_t *sgd_params = (sgd_settings_t*) sgd->settings;
        sgd_params->learning_rate = lrates[i];

        optimizer_run_epoch( sgd, n_train_samples, (float*) train_X->data, (float*) train_Y->data,
                                  0,  NULL, NULL, results+i );
        printf("lr = %f , mse: %e\n", lrates[i], results[i] );

        neuralnet_free( nn );
        neuralnet_t *nn = neuralnet_new( argv[1], split );
        assert( nn );
        neuralnet_set_loss( nn, "binary_crossentropy" );

    }

    strsplit_free(split);
    free(results);
    neuralnet_save( nn, "after-20-epochs.npz" );
    neuralnet_free( nn );
    free( sgd );
    c_npy_matrix_array_free( train_test );
    return 0;
}    
    
