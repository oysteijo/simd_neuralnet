#include "stochastic_gradient_descent.h"
#include "progress.h"
#include "simd.h"
#include <stdlib.h>   /* malloc/free in macros */
#include <stdio.h>    /* fprintf in macro */
#include <string.h>   /* memset */
#include <assert.h>

#include <immintrin.h>


// OPTIMIZER_DEFINE(SGD, newopt->learning_rate = 0.01f; newopt->decay = 0.0f );

/** 
 * Implements a <- a + scale * b   (Which is saxpy, isn't it?)
 * */
/* Hehe - a call to cblas_saxpy instead of this primitive loop, is actually slower!! */
#if defined(__AVX__)
static void apply_update( unsigned int n, float *y, const float scale, const float *b )
{
	unsigned int count = n >> 3;
    unsigned int remaining = n & 0x7;
	const float *b_ptr = b;
	float *y_ptr = y; 

    /* y is always aligned, that is not a problem. However, b, which comes from the gradient is not aligned
       when a n_output from a layer is not mod 8. There is of course a possibility to do things sequensial 
       until we are aligned, and then to vectorized operations, however.... fix this later, it won't gain much. */
    if ( is_aligned( y ) && is_aligned( b ) ){
        __m256 scalevec = _mm256_set1_ps(scale);
        for (int j = count; j; j--, y_ptr += 8, b_ptr += 8){
#if defined(__AVX2__)
            _mm256_store_ps(y_ptr, _mm256_fmadd_ps( _mm256_load_ps(b_ptr), scalevec, _mm256_load_ps(y_ptr)));
#else
            _mm256_store_ps(y_ptr, _mm256_add_ps(_mm256_load_ps(y_ptr), _mm256_mul_ps(_mm256_load_ps(b_ptr), scalevec)));
#endif
        }
    
    } else {
        count = 0;
        remaining = n;
    }

    /* Do the rest. If the user has done his/her homework, this should not be necesarry */
    if( !remaining ) return;

    y_ptr = y + (count*8);
    b_ptr = b + (count*8);

    for ( unsigned int i = 0; i < remaining; i++ )
        *y_ptr++ += scale * *b_ptr++;
}
#else  /* not __AVX__ */
static void apply_update( unsigned int n, float *y, const float scale, const float *b )
{
    float *y_ptr = y;
    const float *b_ptr = b;

    for ( unsigned int i = 0; i < n; i++ )
        *y_ptr++ += scale * *b_ptr++;
}
#endif


/* HERE IS THE SPECIAL SGD CODE */
void SGD_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    sgd_settings_t *sgd = (sgd_settings_t*) opt->settings;

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        /* Batch start */
        memset( opt->batchgrad, 0, n_parameters * sizeof(float));  /* Clear the batch grad */

        int b = 0;
        // printf("batchsize: %d\n", opt->batchsize );
        for ( ; b < opt->batchsize && i < n_train_samples; b++, i++ ){
            neuralnet_backpropagation( nn, train_X + (opt->pivot[i] * n_input), train_Y + (opt->pivot[i] * n_output), opt->grad );
            /* then we add */
            for ( unsigned int w = 0; w < n_parameters; w++ ) /*  Improve this */
                opt->batchgrad[w] += opt->grad[w];
        }
        for ( unsigned int w = 0; w < n_parameters; w++ ) /*  Improve this */
            opt->batchgrad[w] /=  (float) b;
        
        /* OK... */
        if (sgd->decay > 0.0f )
            sgd->learning_rate *= 1.0f / (1.0f + sgd->decay * (float) opt->iterations);
        opt->iterations++;

        /* DISCUSS: This code may be better suited in neuralnet.c */
        const float *ptr = opt->batchgrad;
        for ( int l = 0; l < nn->n_layers; l++ ){
            const int n_inp = nn->layer[l].n_input;
            const int n_out = nn->layer[l].n_output;
            apply_update( n_out, nn->layer[l].bias, -sgd->learning_rate, ptr );
            // cblas_saxpy( n_out, -learning_rate, ptr, 1, nn->layer[l].bias, 1 );
            ptr += n_out;
            apply_update( n_out * n_inp, nn->layer[l].weight, -sgd->learning_rate, ptr );
            // cblas_saxpy( n_out * n_inp, -learning_rate, ptr, 1, nn->layer[l].weight, 1 );
            ptr += n_inp * n_out;
        }
#if 1
        /* Argh! The progress bar only prints out at some values if i (there is a mod operation). This
         * call will therefor do nothing for some values of batchsize. */
        progress_bar("Training: ", i, n_train_samples-1 );
#endif
    }
}
