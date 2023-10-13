#include "evaluate.h"
#include "simd.h"
#include <string.h>

#include <omp.h>

void evaluate( neuralnet_t *nn, const int n_valid_samples, const float *valid_X, const float *valid_Y,
        metric_func metrics[], float *results )
{
    const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    metric_func *mf_ptr = metrics;

    int n_metrics = 0;
    while ( *mf_ptr++ )
        n_metrics++;

    float local_results[n_metrics];
    memset( local_results, 0, n_metrics * sizeof(float));
    #pragma omp parallel for reduction(+:local_results[:])
    for ( int i = 0; i < n_valid_samples; i++ ){
        SIMD_ALIGN(float y_pred[n_output]);
        neuralnet_predict( nn, valid_X + (i*n_input), y_pred );

        float *res = local_results;
        for ( int j = 0; j < n_metrics; j++ ){
            float _error = metrics[j]( n_output, y_pred, valid_Y + (i*n_output));
            *res++ += _error;
        }
    }

    float *res = results;
    for ( int i = 0; i < n_metrics; i++ )
        *res++ = local_results[i] / (float) n_valid_samples;

}
