#include "adagrad.h"
#include "progress.h"
#include <stdlib.h>   /* malloc/free in macros */
#include <stdio.h>    /* fprintf in macro */
#include <string.h>   /* memset */
#include <math.h>
#include <immintrin.h>

static void vector_accumulate( const int n, float *a, const float *b )
{
    int i = 0;
    float *a_ptr = a;
    const float *b_ptr = b;
#ifdef __AVX__
    for ( ; i <= ((n)-8); i += 8, a_ptr += 8, b_ptr += 8 )
        _mm256_store_ps(a_ptr, _mm256_add_ps(_mm256_load_ps(a_ptr), _mm256_load_ps(b_ptr)));
#endif
    for (; i < n; i++ )
        *a_ptr++ += *b_ptr++; 
}

static void vector_divide_by_scalar( const int n, float *v, float scalar )
{
    int i = 0;
    float *v_ptr = v;
#ifdef __AVX__
    __m256 v_scale = _mm256_set1_ps(scalar);
    for ( ; i <= ((n)-8); i += 8, v_ptr += 8)
        _mm256_store_ps(v_ptr, _mm256_div_ps(_mm256_load_ps(v_ptr), v_scale));
#endif
    for( ; i < n; i++ )
        *v_ptr++ /= scalar;
}

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
    const float epsilon = 1.0e-7f;
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

    static bool do_print_settings = true;
    if ( do_print_settings ){
        printf( "learning_rate: %.4f\ndecay: %.4f\n",
            adagrad->learning_rate, adagrad->decay );
        do_print_settings = false;
    }

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        /* Batch start */
        if ( opt->batchsize > 1 )
            memset( opt->batchgrad, 0, n_parameters * sizeof(float));  /* Clear the batch grad */

        int b = 0;
        for ( ; b < opt->batchsize && i < n_train_samples; b++, i++ ){
            neuralnet_backpropagation( nn, train_X + (opt->pivot[i] * n_input), train_Y + (opt->pivot[i] * n_output), opt->grad );
            /* then we add */
            if( opt->batchsize > 1 )
                vector_accumulate( n_parameters, opt->batchgrad, opt->grad );
            /* FIXME: Consider how the progress indicator feedback should be handeled. Maybe as a callback? This seems wrong. */
            progress_bar("Training: ", i, n_train_samples-1 );
        }

        if( opt->batchsize > 1 )
            vector_divide_by_scalar( n_parameters, opt->batchgrad, (float) b );
        
        /* OK... */
        if (adagrad->decay > 0.0f )
            adagrad->learning_rate *= 1.0f / (1.0f + adagrad->decay * (float) opt->iterations);
        opt->iterations++;

        float *delta_w = opt->batchsize > 1 ? opt->batchgrad : opt->grad;
        float *r = opt->velocity; 

        accumulate_squared_gradient( n_parameters, r, delta_w );
        compute_update( n_parameters, delta_w, r, adagrad->learning_rate /*, epsilon? */ );

        neuralnet_update( nn, delta_w );
    }
}
