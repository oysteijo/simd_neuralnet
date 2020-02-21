#include "optimizer_implementations.h"
#include "simd.h"
#include "progress.h"
#include "vector_operations.h"

#include <stdlib.h>   /* malloc/free in macros */
#include <stdio.h>    /* fprintf in macro */
#include <string.h>   /* memset */
#include <math.h>
#include <immintrin.h>

#include <omp.h>

/* Stochastic Gradient Descent */
void SGD_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    sgd_settings_t *sgd = (sgd_settings_t*) opt->settings;
    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        /* Apply interim update */
        if ( sgd->momentum > 0.0f ){
            vector_scale( n_parameters, opt->velocity, sgd->momentum );
            if( sgd->nesterov )  
                neuralnet_update( nn, opt->velocity );
        }

        /* Calculate batch gradient */
        float SIMD_ALIGN(batchgrad[n_parameters]);
        optimizer_calc_batch_gradient( opt, n_train_samples, train_X, train_Y, &i, batchgrad );

        /* Progress callback */
        if( opt->progress) opt->progress( i, n_train_samples, "Train: " );
        
        /* Learning rate update */
        if (sgd->decay > 0.0f )
            sgd->learning_rate *= 1.0f / (1.0f + sgd->decay * (float) opt->iterations);
        opt->iterations++;

        float *neg_eta_grad = batchgrad;
        vector_scale( n_parameters, neg_eta_grad, -sgd->learning_rate );

        if ( sgd->momentum > 0.0f ){
            /* Compute velocity update */
            vector_accumulate( n_parameters, opt->velocity, neg_eta_grad );
        }
         
        /* Apply update */
        if ( sgd->momentum > 0.0f && !sgd->nesterov )  /* if nesterov==true we have already updated based on the velocity */
            neuralnet_update( nn, opt->velocity );
        else
            neuralnet_update( nn, neg_eta_grad );
    }
}

/* RMSprop */

/* FIXME */
/* This static function is shared among RMSprop og Adagrad. Note that I've kept the epsilon outside the sqrt. */
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


void RMSprop_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    RMSprop_settings_t *RMSprop = (RMSprop_settings_t*) opt->settings;

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        /* Apply interim update */
        if ( RMSprop->momentum > 0.0f ){
            vector_scale( n_parameters, opt->velocity, RMSprop->momentum );
            if( RMSprop->nesterov )  
                neuralnet_update( nn, opt->velocity );
        }

        float SIMD_ALIGN(batchgrad[n_parameters]);
        optimizer_calc_batch_gradient( opt, n_train_samples, train_X, train_Y, &i, batchgrad );
        if( opt->progress ) opt->progress( i, n_train_samples, "Train: " );

        
        if (RMSprop->decay > 0.0f )
            RMSprop->learning_rate *= 1.0f / (1.0f + RMSprop->decay * (float) opt->iterations);
        opt->iterations++;

        float *delta_w = batchgrad;
        float *r = opt->r; 

        float SIMD_ALIGN(g2[n_parameters]);
        vector_square_elements( n_parameters, g2, batchgrad );
        vector_saxpby( n_parameters, r, 1.0f - RMSprop->rho, g2, RMSprop->rho );

        compute_update( n_parameters, delta_w, r, RMSprop->learning_rate /*, epsilon? */ );
        if( RMSprop->momentum > 0.0f )
            vector_accumulate( n_parameters, opt->velocity, delta_w );

        if ( RMSprop->momentum > 0.0f && !RMSprop->nesterov )  /* if nesterov==true we have already updated based on the velocity */
            neuralnet_update( nn, opt->velocity );
        else
            neuralnet_update( nn, delta_w );
    }
}

/* Adagrad */

/* This function does  r <- r + g^2 */
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

/* Adam */
/* Discuss: I could do this with vector saxpby where alpha is rho and beta is 1-rho.
 * I'm not sure it will help me anything. */
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

/* Discuss: I could do this with vector saxpby2 where alpha is rho and beta is 1-rho. I just have
 * to handle the g^2 in some way. Then I could call the same function in adagrad (the function now
 * called accumulate_squared_gradient() with alpha set to 1.0 and beta set to 1.0. 
 * It will simplify some of the code, which is good as we might have to vectorize
 * for different architectures. */
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


void adam_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    adam_settings_t *adam = (adam_settings_t*) opt->settings;

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    static float beta_1_corrected = 1.0f;
    static float beta_2_corrected = 1.0f;

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        float SIMD_ALIGN(g[n_parameters]);
        optimizer_calc_batch_gradient( opt, n_train_samples, train_X, train_Y, &i, g );
        if(opt->progress) opt->progress( i, n_train_samples, "Train: " );
        
        opt->iterations++;
        beta_1_corrected *= adam->beta_1;
        beta_2_corrected *= adam->beta_2;

        #pragma omp parallel sections
        {
            #pragma omp section
            update_biased_first_moment ( n_parameters, opt->s, g, adam->beta_1 );
            #pragma omp section
            update_biased_second_moment( n_parameters, opt->r, g, adam->beta_2 );
        }

        compute_update_adam( n_parameters, g, opt->s, opt->r, beta_1_corrected, beta_2_corrected, adam->learning_rate );

        neuralnet_update( nn, g);
    }
}

/* AdamW */
/* FIXME AdamW and Adam is exactly the same thing, just that one can do weight decay and the other not.
 * I think I should get rid of one of them. */
void adamw_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    adamw_settings_t *adamw = (adamw_settings_t*) opt->settings;

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    static float beta_1_corrected = 1.0f;
    static float beta_2_corrected = 1.0f;

    /* One epoch */
    for ( unsigned int i = 0; i < n_train_samples ;  ){

        float SIMD_ALIGN(g[n_parameters]);
        optimizer_calc_batch_gradient( opt, n_train_samples, train_X, train_Y, &i, g );
        if(opt->progress) opt->progress( i, n_train_samples, "Train: " );
        
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

        compute_update_adam( n_parameters, g, opt->s, opt->r, beta_1_corrected, beta_2_corrected, adamw->learning_rate );

        if( adamw->weight_decay > 0.0f ){
            float SIMD_ALIGN(weights[n_parameters]);
            neuralnet_get_parameters( nn, weights );
            vector_saxpy( n_parameters, g, -adamw->weight_decay, weights);
        }

        neuralnet_update( nn, g);
    }
}
