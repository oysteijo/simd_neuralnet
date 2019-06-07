#include "adamw.h"
#include "simd.h"
#include "vector_operations.h"

#include <stdlib.h>   /* malloc/free in macros */
#include <stdio.h>    /* fprintf in macro */
#include <string.h>   /* memset */
#include <math.h>
#include <immintrin.h>

#include <omp.h>

static void get_weights( const neuralnet_t *nn, float *weights )
{
    float *ptr = weights;
    for ( int l = 0; l < nn->n_layers; l++ ){
        const int n_inp = nn->layer[l].n_input;
        const int n_out = nn->layer[l].n_output;
        memcpy( ptr, nn->layer[l].bias, n_out * sizeof(float) );
        ptr += n_out;
        memcpy( ptr, nn->layer[l].weight, n_out * n_inp * sizeof(float) );
        ptr += n_inp * n_out;
    }
}

static void update_biased_first_moment( const int n , float *s, const float *g, const float rho )
{
    int i = 0;
    float *s_ptr = s;
    const float *g_ptr = g;
#ifdef __AVX__
    __m256 rhov = _mm256_set1_ps( rho );
    __m256 one_minus_rhov = _mm256_set1_ps( 1.0f - rho );
    for( ; i <= ((n)-8); i += 8 , s_ptr += 8, g_ptr += 8 ){
        __m256 gv = _mm256_load_ps( g_ptr );
        _mm256_store_ps( s_ptr,
                _mm256_add_ps(
                    _mm256_mul_ps( _mm256_load_ps( s_ptr ), rhov ),
                    _mm256_mul_ps( gv, one_minus_rhov )
                    )
                );
    }
#endif
    for( ; i < n; i++, s_ptr++ ){
        const float gval = g[i];
        *s_ptr = (rho * *s_ptr) + (1.0f - rho) * gval;
    }
}

static void update_biased_second_moment( const int n, float *r, const float *g, const float rho )
{
    int i = 0;
    float *r_ptr = r;
    const float *g_ptr = g;
#ifdef __AVX__
    __m256 rhov = _mm256_set1_ps( rho );
    __m256 one_minus_rhov = _mm256_set1_ps( 1.0f - rho );
    for( ; i <= ((n)-8); i += 8 , r_ptr += 8, g_ptr += 8 ){
        __m256 gv = _mm256_load_ps( g_ptr );
        _mm256_store_ps( r_ptr,
                _mm256_add_ps(
                    _mm256_mul_ps( _mm256_load_ps( r_ptr ), rhov ),
                    _mm256_mul_ps( _mm256_mul_ps( gv, gv ), one_minus_rhov )
                    )
                );
    }
#endif
    for( ; i < n; i++, r_ptr++ ){
        const float gval = g[i];
        *r_ptr = (rho * *r_ptr) + (1.0f - rho) * gval * gval;
    }
}

static void compute_update( const int n, float *delta_w, const float *s, const float *r, const float rho1, const float rho2, const float lr )
{
    const float epsilon = 1.0e-8f;
    int i = 0;
    const float *r_ptr = r;
    const float *s_ptr = s;
    const float one_minus_rho1 = 1.0f - rho1;
    const float one_minus_rho2 = 1.0f - rho2;
#ifdef __AVX__
    const __m256 one_minus_rho1_v = _mm256_set1_ps(one_minus_rho1);
    const __m256 one_minus_rho2_v = _mm256_set1_ps(one_minus_rho2);
    const __m256 lr_v = _mm256_set1_ps(-lr);
    const __m256 eps_v = _mm256_set1_ps(epsilon);

    for( ; i <= ((n)-8) ; i += 8, delta_w += 8, s_ptr += 8, r_ptr += 8 ){
        const __m256 s_hat = _mm256_div_ps( _mm256_load_ps( s_ptr ), one_minus_rho1_v );
        const __m256 r_hat = _mm256_div_ps( _mm256_load_ps( r_ptr ), one_minus_rho2_v );

        _mm256_store_ps( delta_w, 
                _mm256_div_ps(  _mm256_mul_ps ( lr_v, s_hat ), _mm256_add_ps ( _mm256_sqrt_ps( r_hat ), eps_v ) ));
    }
#endif
    for( ; i < n; i++){
        const float s_hat = *s_ptr++ / one_minus_rho1;
        const float r_hat = *r_ptr++ / one_minus_rho2;
        *delta_w++ = -lr * s_hat / (sqrtf( r_hat ) + epsilon);
    }
}


void adamw_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    adamw_settings_t *adamw = (adamw_settings_t*) opt->settings;

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    static float beta_1_corrected = 1.0f;
    static float beta_2_corrected = 1.0f;

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        float SIMD_ALIGN(batchgrad[n_parameters]);
        memset( batchgrad, 0, n_parameters * sizeof(float));  /* Clear the batch grad */

        int remaining_samples = (int) n_train_samples - (int) i;
        int max_loop = remaining_samples < opt->batchsize ? remaining_samples : opt->batchsize;
        #pragma omp parallel for shared(i) reduction(+:batchgrad[:])
        for ( int b = 0 ; b < max_loop; b++){
            float SIMD_ALIGN(grad[n_parameters]);
            neuralnet_backpropagation( nn, train_X + (opt->pivot[i] * n_input), train_Y + (opt->pivot[i] * n_output), grad );
            vector_accumulate( n_parameters, batchgrad, grad );
            #pragma omp atomic update
            i++;
        }
        vector_divide_by_scalar( n_parameters, batchgrad, (float) max_loop );
        opt->progress( i, n_train_samples, "Train: " );
        
        float *g = batchgrad;

        opt->iterations++;
        beta_1_corrected *= adamw->beta_1;
        beta_2_corrected *= adamw->beta_2;

        #pragma omp parallel sections
        {
            #pragma omp section
            update_biased_first_moment ( n_parameters, opt->s, g, adamw->beta_1 );
            #pragma omp section
            update_biased_second_moment( n_parameters, opt->r, g, adamw->beta_2 );
        }

        compute_update( n_parameters, g, opt->s, opt->r, beta_1_corrected, beta_2_corrected, adamw->learning_rate );

        float SIMD_ALIGN(weights[n_parameters]);
        get_weights( nn, weights );
#if 0
        vector_scale( n_parameters, weights, -adamw->weight_decay );
        vector_accumulate( n_parameters, g, weights );
#endif
        vector_saxpy( n_parameters, g, -adamw->weight_decay, weights);

        neuralnet_update( nn, g);
    }
}
