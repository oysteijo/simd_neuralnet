#include "stochastic_gradient_descent.h"
#include "progress.h"
#include "simd.h"
#include <stdlib.h>   /* malloc/free in macros */
#include <stdio.h>    /* fprintf in macro */
#include <string.h>   /* memset */
#include <assert.h>

#include <immintrin.h>


OPTIMIZER_DEFINE(SGD, newopt->learning_rate = 0.01f; newopt->decay = 0.0f );

/* FIXME optimizer common code */
/* Hmmm ? Shouldn't this function rather take in a optimizer? Or maybe not? Maybe it should be independent of optimizer? */
static void evaluate( neuralnet_t *nn, const int n_valid_samples, const float *valid_X, const float *valid_Y,
        metric_func metrics[], float *results )
{
    const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    metric_func *mf_ptr = metrics;

    int n_metrics = 0;
    while ( *mf_ptr++ )
        n_metrics++;

    memset( results, 0, n_metrics * sizeof(float)); // float total_error = 0.0f;

    for ( int i = 0; i < n_valid_samples; i++ ){
        float y_pred[n_output];
        neuralnet_predict( nn, valid_X + (i*n_input), y_pred );

        float *res = results;
        for ( int j = 0; j < n_metrics; j++ ){
            float _error = metrics[j]( n_output, y_pred, valid_Y + (i*n_output));
            *res++ += _error;
        }
    }

    float *res = results;
    for ( int i = 0; i < n_metrics; i++ )
        *res++ /= (float) n_valid_samples;

    res = results;
    for ( int i = 0; i < n_metrics; i++ )
        printf( "%s: %5.5e  ", get_metric_name( metrics[i] ), *res++ );

}

static void fisher_yates_shuffle( unsigned int n, unsigned int *arr )
{
    for ( unsigned int i = n-1; i > 0; i-- ){
        unsigned int j = rand() % (i+1);
        unsigned int tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

/** 
 * Implements a <- a + scale * b   (Which is saxpy, isn't it?)
 * */
/* Hehe - a call to cblas_saxpy instead of this primitive loop, is actually slower!! */
#if defined(__AVX__)
void apply_update( unsigned int n, float *y, const float scale, const float *b )
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
void apply_update( unsigned int n, float *y, const float scale, const float *b )
{
    float *y_ptr = y;
    const float *b_ptr = b;

    for ( unsigned int i = 0; i < n; i++ )
        *y_ptr++ += scale * *b_ptr++;
}
#endif



/* HERE IS THE SPECIAL SGD CODE */
static void SGD_run_epoch( const optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y,
        const unsigned int n_valid_samples, const float *valid_X, const float *valid_Y, float *results )
{
    SGD_t *sgd = (SGD_t*) opt;
    neuralnet_t *nn = sgd->opt.nn;

    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    /* FIXME: I should rather have an array of pointers instead of an array of integers --
      ... or maybe not? Then I have to shuffle in the same order both the trainX and train_Y ... */
    unsigned int *pivot = malloc( n_train_samples * sizeof(unsigned int) );
    if ( !pivot ){
        fprintf( stderr, "Cannot allocate pivot array.\n");
        return;
    }
    for ( unsigned int i = 0; i < n_train_samples; i++ )
        pivot[i] = i;
#if 0
    if( sgd->opt.shuffle )
        fisher_yates_shuffle( n_train_samples, pivot );
#endif

    float *train_X_ptr = (float*) train_X;
    float *train_Y_ptr = (float*) train_Y;

    float *batch_grad = simd_malloc(n_parameters * sizeof(float)); /* simd? */
    if( !batch_grad ){
        printf("ERROR: Cannot allocate memory for batch gradients\n");
        free( pivot );
        return;
    }

    float *grad       = simd_malloc(n_parameters * sizeof(float)); /* simd? */
    if( !batch_grad ){
        printf("ERROR: Cannot allocate memory for sample gradients\n");
        free( batch_grad );
        free( pivot );
        return;
    }

    /* Epoch start */
    for ( unsigned int i = 0; i < n_train_samples ;  ){

        /* Batch start */
        memset( batch_grad, 0, n_parameters * sizeof(float));  /* Clear the batch grad */

        int b = 0;
        for ( ; b < opt->batchsize && i < n_train_samples; b++, i++ ){
            neuralnet_backpropagation( nn, train_X_ptr + (pivot[i] * n_input), train_Y_ptr + (pivot[i] * n_output), grad );
            /* then we add */
            for ( unsigned int w = 0; w < n_parameters; w++ ) /*  Improve this */
                batch_grad[w] += grad[w];
        }
        assert ( b == 1 );
        for ( unsigned int w = 0; w < n_parameters; w++ ) /*  Improve this */
            batch_grad[w] /=  (float) b;
        
        /* OK... */
        if (sgd->decay > 0.0f )
            sgd->learning_rate *= 1.0f / (1.0f + sgd->decay * (float) sgd->opt.iterations);
        sgd->opt.iterations++;

        const float *ptr = batch_grad;
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
        progress_bar("Training: ", i, n_train_samples-1 );
#endif
    }
    free( pivot );
    free( grad );
    free( batch_grad );

    /* This is not about the SGD, and should be general. This code belongs in optimizer.c. I need someting like a
       an abstract class in a template pattern. */

    /* Calculate the losses */
    /* First the train loss */
    int n_metrics = optimizer_get_n_metrics( opt );
    evaluate( nn, n_train_samples, train_X, train_Y, OPTIMIZER(sgd)->metrics, results );  

    /* and if validation is given - do it */
    if (valid_X && valid_Y && n_valid_samples > 0 ){
        evaluate( nn, n_valid_samples, valid_X, valid_Y, OPTIMIZER(sgd)->metrics, results + n_metrics );
        printf("\n");
    }

#if 0
    /* callbacks... Just typing something, it's a code sketch */
    if ( sgd->opt.n_callbacks > 0 )
        for( cb = 0; cb < sgd->opt.n_callbacks; cb++)
            sgd->opt.callback[cb]( OPTIMIZER( sgd ), train_X, train_Y, test_X, test_Y, batch_size );
#endif
//    return 0.0f;
} 

