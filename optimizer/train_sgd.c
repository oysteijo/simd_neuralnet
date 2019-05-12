#include "stochastic_gradient_descent.h"
#include "c_npy.h"
#include "neuralnet.h"
#include "activation.h"
#include "loss.h"
#include "metrics.h"

#include "strtools.h"
#include "progress.h"

#include <cblas.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "simd.h"
#include <immintrin.h>


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
    strsplit_free(split);
    neuralnet_set_loss( nn, "mean_squared_error" );


    optimizer_t *sgd = OPTIMIZER(SGD_new( nn, NULL ));
    


    int n_epochs = 1;
    srand( 70 );
    float *results = calloc( 2 * n_epochs, sizeof(float));
    
    for ( int epoch = 0; epoch < n_epochs; epoch++ ){

        optimizer_run_epoch( sgd, n_train_samples, (float*) train_X->data, (float*) train_Y->data,
                                  n_test_samples,  (float*) test_X->data, (float*) test_Y->data , results + 2*epoch);

    }

    printf(" mse: %e  , mse: %e\n", results[0], results[1]);

    free(results);
    /* log and report */
    neuralnet_save( nn, "after-20-epochs.npz" );
    neuralnet_free( nn );
    c_npy_matrix_array_free( train_test );
    return 0;
}    
    
