#include "stochastic_gradient_descent.h"
#include "progress.h"
#include <stdlib.h>   /* malloc/free in macros */
#include <stdio.h>    /* fprintf in macro */
#include <string.h>   /* memset */
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

static void vector_scale( const int n, float *v, float scalar )
{
    int i = 0;
    float *v_ptr = v;
#ifdef __AVX__
    __m256 v_scale = _mm256_set1_ps(scalar);
    for ( ; i <= ((n)-8); i += 8, v_ptr += 8)
        _mm256_store_ps(v_ptr, _mm256_mul_ps(_mm256_load_ps(v_ptr), v_scale));
#endif
    for( ; i < n; i++ )
        *v_ptr++ *= scalar;
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

/* HERE IS THE SPECIAL SGD CODE */
void SGD_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    sgd_settings_t *sgd = (sgd_settings_t*) opt->settings;

    static bool do_print_settings = true;
    if ( do_print_settings ){
        printf( "learning_rate: %.4f\nmomentum: %.4f\ndecay: %.4f\nNesterov: %s\n",
            sgd->learning_rate, sgd->momentum, sgd->decay, sgd->nesterov ? "True" : "False");
        do_print_settings = false;
    }

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        if ( sgd->momentum > 0.0f ){
            vector_scale( n_parameters, opt->velocity, sgd->momentum );
            if( sgd->nesterov )  
                neuralnet_update( nn, opt->velocity );
        }
        /* Batch start */
        if ( opt->batchsize > 1 )
            memset( opt->batchgrad, 0, n_parameters * sizeof(float));  /* Clear the batch grad */

        int b = 0;
        for ( ; b < opt->batchsize && i < n_train_samples; b++, i++ ){
            neuralnet_backpropagation( nn, train_X + (opt->pivot[i] * n_input), train_Y + (opt->pivot[i] * n_output), opt->grad );
            /* then we add */
            if( opt->batchsize > 1 )
                vector_accumulate( n_parameters, opt->batchgrad, opt->grad );
        }
        opt->progress( i, n_train_samples, "Train: " );

        if( opt->batchsize > 1 )
            vector_divide_by_scalar( n_parameters, opt->batchgrad, (float) b );
        
        /* OK... */
        if (sgd->decay > 0.0f )
            sgd->learning_rate *= 1.0f / (1.0f + sgd->decay * (float) opt->iterations);
        opt->iterations++;

        float *neg_eta_grad = opt->batchsize > 1 ? opt->batchgrad : opt->grad;
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
