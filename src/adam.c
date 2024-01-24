#include "adam.h"

#include "simd.h" 
#include "matrix_operations.h" 

#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __AVX__ 
#include <immintrin.h>
#endif

/*  adam.c  */
struct _adam_t 
{
    optimizer_t opt;
    /* Other data */
    float learning_rate;
    float beta_1;
    float beta_2;
    float weight_decay;

    /* private stuff - don't touch! */
    float *r;
    float *s;
};

static void adam_optimizer_init( adam_t *adam, adam_properties_t *properties )
{
    adam_properties_t *props = (adam_properties_t*) properties;

    adam->learning_rate = props->learning_rate;
    adam->beta_1 = props->beta_1;
    adam->beta_2 = props->beta_2;
    adam->weight_decay = props->weight_decay;

    const unsigned int n_param = neuralnet_total_n_parameters( OPTIMIZER(adam)->nn );

    adam->r   = simd_malloc( n_param * sizeof(float) );
    adam->s   = simd_malloc( n_param * sizeof(float) );
    assert( adam->r );
    assert( adam->s );
    memset( adam->r, 0, n_param * sizeof(float));
    memset( adam->s, 0, n_param * sizeof(float));
}

static void adam_optimizer_free( optimizer_t *opt )
{
    if( !opt ) return;
    simd_free( ADAM_OPTIMIZER(opt)->r );
    simd_free( ADAM_OPTIMIZER(opt)->s );
}

OPTIMIZER_DEFINE(adam, 
    adam_optimizer_init( newopt, properties );
    newopt->opt.free = adam_optimizer_free;
);

/* Adam */
/* Discuss: I could do this with vector saxpby where alpha is rho and beta is 1-rho.
 * I'm not sure it will help me anything. */
static void update_biased_first_moment( const int n , float *s, const float *g, const float rho )
{
    int i = 0;
    float *s_ptr = s;
#ifdef __AVX__
    const float *g_ptr = g;
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

/* Discuss: I could do this with vector saxpby2 where alpha is rho and beta is 1-rho. I just have
 * to handle the g^2 in some way. Then I could call the same function in adagrad (the function now
 * called accumulate_squared_gradient() with alpha set to 1.0 and beta set to 1.0. 
 * It will simplify some of the code, which is good as we might have to vectorize
 * for different architectures. */
static void update_biased_second_moment( const int n, float *r, const float *g, const float rho )
{
    int i = 0;
    float *r_ptr = r;
#ifdef __AVX__
    const float *g_ptr = g;
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

/* FIXME */
static void compute_update_adam( const int n, float *delta_w, const float *s, const float *r, const float rho1, const float rho2, const float lr )
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

/* This is adding the Decoupled Weight Decay Regulatization suggested by
 * by Ilya Loshchilov, Frank Hutter (2019) aka. AdamW */
void adam_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    // adamw_settings_t *adamw = (adamw_settings_t*) opt->settings;

    adam_t *adam = ADAM_OPTIMIZER(opt);
    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    static float beta_1_corrected = 1.0f;
    static float beta_2_corrected = 1.0f;

    /* One epoch */
    for ( unsigned int i = 0; i < n_train_samples ;  ){

        float SIMD_ALIGN(g[n_parameters]);
        optimizer_calc_batch_gradient( opt, n_train_samples, train_X, train_Y, &i, g );
        if(opt->progress) opt->progress( i, n_train_samples, "Train: " );
        
        beta_1_corrected *= adam->beta_1;
        beta_2_corrected *= adam->beta_2;

        #pragma omp parallel sections
        {
            #pragma omp section
            update_biased_first_moment ( n_parameters, adam->s, g, adam->beta_1 );
            #pragma omp section
            update_biased_second_moment( n_parameters, adam->r, g, adam->beta_2 );
        }

        compute_update_adam( n_parameters, g, adam->s, adam->r, beta_1_corrected, beta_2_corrected, adam->learning_rate );

        if( adam->weight_decay > 0.0f ){
            float SIMD_ALIGN(weights[n_parameters]);
            neuralnet_get_parameters( nn, weights );
            vector_saxpy( n_parameters, g, -adam->weight_decay, weights);
        }

        neuralnet_update( nn, g);
    }
}

