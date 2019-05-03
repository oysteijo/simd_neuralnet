#include "neuralnet.h"
#include "metrics.h"
#include "c_npy.h"

#include "strtools.h"

#include <stdio.h>
#include <assert.h>

float evaluate( neuralnet_t *nn, cmatrix_t *test_X, cmatrix_t *test_Y, metric_func metric )
{
    float *test_X_ptr = (float*) test_X->data;
    float *test_Y_ptr = (float*) test_Y->data;

    const size_t n_test_samples = test_X->shape[0];
    assert( n_test_samples == test_Y->shape[0] );

    const int n_input = test_X->shape[1];
    const int n_output = test_Y->shape[1];

    /* if any of these asserts fails, try to open the weight in python and save with
     * np.ascontiguousarray( matrix ) */
    assert( test_X->fortran_order == false );
    assert( test_Y->fortran_order == false );

    /* OpenMP thread */
    float total_error = 0.0f;
    for ( unsigned int i = 0; i < n_test_samples; i++ ){
        float y_pred[n_output];
        neuralnet_predict( nn, test_X_ptr + (i*n_input), y_pred );
        total_error += metric( n_output, y_pred, test_Y_ptr + (i*n_output));
    }
    return total_error /= (float) n_test_samples;
}

STRSPLIT_INIT
STRSPLIT_LENGTH_INIT
STRSPLIT_FREE_INIT

int main( int argc, char *argv[] )
{
    if (argc != 5 ){
        fprintf( stderr, "Usage: %s <weightsfile.npz> <quoted list of activations> <trainsamples.npz> <metric_func>\n", argv[0] );
        return 0;
    }

    cmatrix_t **train_test = c_npy_matrix_array_read( argv[3]);
    if( !train_test ) return -1;

    /* FIXME : This assumes there are four arrays in the npz file -- it can be any even number! */
    cmatrix_t *train_X = train_test[0];
    cmatrix_t *train_Y = train_test[1];
    cmatrix_t *test_X = train_test[2];
    cmatrix_t *test_Y = train_test[3];

    assert( test_X->fortran_order == false );
    assert( test_Y->fortran_order == false );

    char **split = strsplit( argv[2], ',' );
    int len_split = strsplit_length( split );
    for ( int i = 0; i < len_split; i++ ){
        strstrip( split[i] );
    }
#if 0
    for ( int i = 0; i < len_split; i++ ){
        printf("%d:%s\n", i, split[i] );
    }
#endif
    neuralnet_t *nn = neuralnet_new( argv[1], split );
    assert( nn );
    strsplit_free(split);

    metric_func metric = get_metric_func( argv[4] );
    printf("Train metric (%s): %.7f\n", get_metric_name( metric ), evaluate( nn, train_X, train_Y, metric ));
    printf(" Test metric (%s): %.7f\n", get_metric_name( metric ), evaluate( nn, test_X, test_Y, metric ));

    /* cleanup */
    c_npy_matrix_array_free( train_test );
    neuralnet_free(nn);
    return 0;
}
