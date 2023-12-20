#include "npy_array.h"
#include "npy_array_list.h"
#include "neuralnet.h"
#include "neuralnet_predict_batch.h"
#include "simd.h"

#include "evaluate.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
void local_evaluate( neuralnet_t *nn, const int n_valid_samples, const float *valid_X, const float *valid_Y,
        metric_func metrics[], float *results )
{
    //const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    metric_func *mf_ptr = metrics;

    int n_metrics = 0;
    while ( *mf_ptr++ )
        n_metrics++;

    float predictions[ n_output * n_valid_samples ];
    neuralnet_predict_batch( nn, n_valid_samples, valid_X, predictions);

    float local_results[n_metrics];
    memset( local_results, 0, n_metrics * sizeof(float));
    for ( int i = 0; i < n_valid_samples; i++ ){
        float *res = local_results;
        for ( int j = 0; j < n_metrics; j++ ){
            float _error = metrics[j]( n_output, predictions + (i*n_output), valid_Y + (i*n_output));
            *res++ += _error;
        }
    }

    float *res = results;
    for ( int i = 0; i < n_metrics; i++ )
        *res++ = local_results[i] / (float) n_valid_samples;
}

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
    neuralnet_t *nn = neuralnet_load( "mushroom-neuralnet.npz");
    assert( nn );

    const int n_train_samples = train_X->shape[0];
    const int n_test_samples = test_X->shape[0];

    metric_func metrics[] = { get_metric_func("binary_crossentropy"), get_metric_func( "binary_accuracy"), NULL };

    float results[ 2 * 2 ];
    local_evaluate( nn, n_train_samples, (float*) train_X->data, (float*) train_Y->data, metrics, results);
    local_evaluate( nn, n_test_samples, (float*) test_X->data, (float*) test_Y->data, metrics, results + 2);

    printf("Train loss    : %5.5f\n", results[0] );
    printf("Train accuracy: %5.5f\n", results[1] );
    printf("Test loss     : %5.5f\n", results[2] );
    printf("Test accuracy : %5.5f\n", results[3] );

    /* Clean up the resources */
    neuralnet_free( nn );
    npy_array_list_free( filelist );
    return 0;
}

