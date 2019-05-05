#include "optimizer.h"
#include "simd.h"
#include <stdlib.h>   /* malloc/free in macros */
#include <stdio.h>    /* fprintf in macro */
#include <string.h>   /* memset */
#include <assert.h>

OPTIMIZER_DECLARE(SGD);

struct _SGD_t {
   optimizer_t opt;
   /* other data */
   float learning_rate;
   float decay;
   /* float momentum; */
   /* bool nesterov; */
};

OPTIMIZER_DEFINE(SGD);

static float SDG_run_epoch( const optimizer_t *opt, const cmatrix_t *train_X, const cmatrix_t *train_Y, 
                                       const cmatrix_t *test_X, const cmatrix_t *test_Y, int batchsize )
{
    SGD_t *sgd = (SGD_t*) opt;
    neuralnet_t *nn = sgd->opt.nn;

    unsigned int n_train_samples = train_X->shape[0];
    unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    const int n_input  = train_X->shape[1];
    const int n_output = train_Y->shape[1];

    assert( n_input == nn->layer[0].n_input );
    assert( n_output == nn->layer[nn->n_layers-1].n_output );


    unsigned int *pivot = malloc( n_train_samples * sizeof(unsigned int) );
    if ( !pivot ){
        fprintf( stderr, "Cannot allocate pivot array.\n");
        return 0;
    }
    for ( unsigned int i = 0; i < n_train_samples; i++ )
        pivot[i] = i;

    if( sgd->opt.shuffle )
        fisher_yates_shuffle( n_train_samples, pivot );

    float *train_X_ptr = (float*) train_X->data;
    float *train_Y_ptr = (float*) train_Y->data;
    for ( unsigned int i = 0; i < n_train_samples ;  ){

        float SIMD_ALIGN(batch_grad[n_parameters]); /* simd? */
        memset( batch_grad, 0, n_parameters * sizeof(float));

        int b = 0;
        for ( ; b < batchsize && i < n_train_samples; b++, i++ ){
            float SIMD_ALIGN(grad[n_parameters]); /* simd? */
            neuralnet_backpropagation( nn, train_X_ptr + (pivot[i] * n_input), train_Y_ptr + (pivot[i] * n_output), grad );
            /* then we add */
            for ( unsigned int w = 0; w < n_parameters; w++ ) /*  Improve this */
                batch_grad[w] += grad[w];
        }
        for ( unsigned int w = 0; w < n_parameters; w++ ) /*  Improve this */
            batch_grad[w] /=  b + 1.0f;
        
        /* OK... */
        if (sgd->decay > 0.0f )
            sgd->learning_rate *= 1.0f / (1.0f + sgd->decay * (float) sgd->opt.iterations);
        sgd->opt.iterations++;

        const float *ptr = batch_grad;
        for ( int l = 0; l < nn->n_layers; l++ ){
            const int n_inp = nn->layer[l].n_input;
            const int n_out = nn->layer[l].n_output;
            scale_and_add_vector( n_out, nn->layer[l].bias, -sgd->learning_rate, ptr );
            // cblas_saxpy( n_out, -learning_rate, ptr, 1, nn->layer[l].bias, 1 );
            ptr += n_out;
            scale_and_add_vector( n_out * n_inp, nn->layer[l].weight, -sgd->learning_rate, ptr );
            // cblas_saxpy( n_out * n_inp, -learning_rate, ptr, 1, nn->layer[l].weight, 1 );
            ptr += n_inp * n_out;
        }
    }
    free( pivot );
#if 0
    /* callbacks... Just typing something, it's a code sketch */
    if ( sgd->opt.n_callbacks > 0 )
        for( cb = 0; cb < sgd->opt.n_callbacks; cb++)
            sgd->opt.callback[cb]( OPTIMIZER( sgd ), train_X, train_Y, test_X, test_Y, batch_size );

    /* Test */
    if (test_X && test_Y )
        evaluate( ... );
    else 
        evaluate
#endif
    return 0.0f;
} 

