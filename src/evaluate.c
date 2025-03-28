/* evaluate.c - Øystein Schønning-Johansen 2013 - 2023 */
/* 
 vim: ts=4 sw=4 softtabstop=4 expandtab 
*/
#include "evaluate.h"
#include "neuralnet_predict_batch.h"
#include "simd.h"
#include <string.h>

#include <omp.h>

void evaluate( neuralnet_t *nn, const int n_valid_samples, const float *valid_X, const float *valid_Y,
        metric_func metrics[], float *results )
{
#ifndef USE_CBLAS 
    const int n_input  = nn->layer[0].n_input;
#endif
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    metric_func *mf_ptr = metrics;

    int n_metrics = 0;
    while ( *mf_ptr++ ){
        n_metrics++;
	}

	if( n_metrics == 0 ){
		*results = -1.0f;
		return;
	}

    float local_results[n_metrics];
    memset( local_results, 0, n_metrics * sizeof(float));
#ifdef USE_CBLAS
    float predictions[ n_output * n_valid_samples ];
    memset( predictions, 0, n_output * n_valid_samples * sizeof(float));
    neuralnet_predict_batch( nn, n_valid_samples, valid_X, predictions);
#endif
    #pragma omp parallel for reduction(+:local_results[:])
    for ( int i = 0; i < n_valid_samples; i++ ){
#ifdef USE_CBLAS
        float *y_pred = predictions + (i*n_output);
#else
        SIMD_ALIGN(float y_pred[n_output]);
        neuralnet_predict( nn, valid_X + (i*n_input), y_pred );
#endif
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
