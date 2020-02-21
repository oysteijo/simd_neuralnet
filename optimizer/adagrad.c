#include "adagrad.h"
#include "progress.h"
#include "vector_operations.h"
#include "simd.h"

#include <stdlib.h>   /* malloc/free in macros */
#include <stdio.h>    /* fprintf in macro */
#include <string.h>   /* memset */
#include <math.h>
#include <immintrin.h>

static void accumulate_squared_gradient( const int n, float *r, const float *g )
{
    int i = 0;
    float *r_ptr = r;
    const float *g_ptr = g;
#ifdef __AVX__
    for( ; i <= ((n)-8); i += 8 , r_ptr += 8, g_ptr += 8 ){
        __m256 gv = _mm256_load_ps( g_ptr );
        _mm256_store_ps( r_ptr, _mm256_add_ps( _mm256_load_ps( r_ptr ), _mm256_mul_ps( gv, gv ) ) );
    }
#endif
    for( ; i < n; i++ ){
        const float gval = g[i];
        *r_ptr++ += gval * gval;
    }
}

static void compute_update( const int n, float *delta_w, const float *r, const float lr )
{
    const float epsilon = 1.0e-7f; /* Same as default K.epsilon (when backend=TensorFlow) in Keras */
    int i = 0;
    const float *r_ptr = r;
#ifdef __AVX__
    const __m256 lr_v = _mm256_set1_ps(-lr);
    const __m256 eps_v = _mm256_set1_ps(epsilon);
    for( ; i <= ((n)-8) ; i += 8, delta_w += 8, r_ptr += 8 ){
        _mm256_store_ps( delta_w, _mm256_mul_ps( _mm256_load_ps( delta_w ),
                    _mm256_div_ps( lr_v, _mm256_add_ps( eps_v, _mm256_sqrt_ps( _mm256_load_ps( r_ptr ) )))));
    }
#endif
    for( ; i < n; i++)
        *delta_w++ *= -lr / ( epsilon + sqrtf( *r_ptr++ ));
}


void adagrad_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    adagrad_settings_t *adagrad = (adagrad_settings_t*) opt->settings;

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        float SIMD_ALIGN(batchgrad[n_parameters]);
        optimizer_calc_batch_gradient( opt, n_train_samples, train_X, train_Y, &i, batchgrad );
        if( opt->progress ) opt->progress( i, n_train_samples, "Train: " );
        
        if (adagrad->decay > 0.0f )
            adagrad->learning_rate *= 1.0f / (1.0f + adagrad->decay * (float) opt->iterations);
        opt->iterations++;

        float *delta_w = batchgrad;
        float *r = opt->velocity; 

        accumulate_squared_gradient( n_parameters, r, delta_w );
        compute_update( n_parameters, delta_w, r, adagrad->learning_rate /*, epsilon? */ );

        neuralnet_update( nn, delta_w );
    }
}