/* neuralnet_predict_batch.c - Øystein Schønning-Johansen 2023 */
/* 
 vim: ts=4 sw=4 softtabstop=4 expandtab 
*/
#include "neuralnet_predict_batch.h"
#include "activation.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cblas.h>
#include <assert.h>

#ifndef USE_CBLAS
/* This is the primitive implemetation using OpenMP to thread th foward calculation of several samples.
 * The recommendation is to us the BLAS implementation, and then add the threading at a higher level in
 * you application. */
void neuralnet_predict_batch( const neuralnet_t *nn, const int n_samples, const float *inputs, float *output )
{
    const int n_inputs = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;
#pragma omp parallel for
    for ( int i = 0; i < n_samples; i++ )
        neuralnet_predict( nn, inputs + i*n_inputs, output + i*n_output);
}
#else
/* This number depends on your system - how much memory do you want to stack allocate?
 * On a desktop or laptop you probably have plenty. If you ever run into a stack overflow,
 * you can recomile with -DN_STACK_ALLOC_FLOATS=1024 (or even a lower number)
 *
 * So this function, `neuralnet_predict_batch()`, uses some work memory. I don't want to
 * allocate this dynamically as heap allocation will be performance killer, and stack 
 * allocation can blow up the stack. The idea is therefore to allocate (on stack) some
 * fixed size memory, N_STACK_ALLOC_FLOATS, and then check if we need mor or less than
 * this. If the size need is more than the allocated, we simply split the set in two
 * and recurse like a divide-and-conquere scheme.
 */
#ifndef N_STACK_ALLOC_FLOATS
#define N_STACK_ALLOC_FLOATS 64 * 1024
#endif

void neuralnet_predict_batch( const neuralnet_t *nn, const int n_samples, const float *inputs, float *output )
{
    /* Make some work memory on stack. First calculate how much we need. */
    int workmem_sz = 0;
    for( int i = 0; i < nn->n_layers; i++)
        workmem_sz += nn->layer[i].n_output;
    workmem_sz *= n_samples;  /* This size is also in floats */

#if 0
    /* Let's see how often this this fails. */
    assert( N_STACK_ALLOC_FLOATS >= workmem_sz && "Stack size limit reached - "
            "either recompile with a higher limit find another way to handle work memory" );
#endif

    if( N_STACK_ALLOC_FLOATS < workmem_sz ){
        int half = n_samples >> 1;
        const int n_inputs = nn->layer[0].n_input;
        const int n_output = nn->layer[nn->n_layers-1].n_output;
        // fprintf(stderr, "Warning: Stack limit reached with %d samples - recursing.\n", n_samples);
        neuralnet_predict_batch( nn, half, inputs, output );
        neuralnet_predict_batch( nn, n_samples - half, inputs+(half*n_inputs), output+(half*n_output) );
        return;    
    }

    /* So the above line makes sure we don't blow off the stack - but what to do when we hit the limit?
     *
     * BTW: What is the limit here? The stack is usually about one MB, so I have initially
     * set the limit to 512 kb. If we have a neural net with say 1000 n_ouptus (cumulative over
     * all layers) and then n_samples is 600... That actually fucks it up.
     *
     * Options:
     * 1. Divide and Conquere: Divide the output into two halfs and recurse. Cool!
     * 2. Have a private (static) function that controls memory. Say:
     *
     *     `float *workmem = _get_workmemory( nn, n_samples );`
     *
     *    and this function returns a pointer to a preallocated area of memory.
     * 3. Have a pointer input for work memory and let the caller take care.
     * 4. Allocate on heap .... ?
     *
     * ... and then: even if I have the above limit, I can still get stack overflow.
     *
     * (so far we stay with option 1, but we allocate a fixed size on the stack suck that it
     * cannot overflow.)
     * */

    float workmem[ N_STACK_ALLOC_FLOATS ]; /* can we blow the stack here? */
    float *activations[nn->n_layers+1];
    activations[0] = (float*) inputs;
    activations[1] = workmem;

    for( int i = 1; i < nn->n_layers-1; i++)
        activations[i+1] = activations[i] + nn->layer[i-1].n_output * n_samples;

    activations[nn->n_layers] = output;

    /* Oh, I have to fill the activations with the biases */
    for( int i = 0; i < nn->n_layers; i++){
        const layer_t *layer_ptr = nn->layer + i;
        const size_t size = layer_ptr->n_output * sizeof(float);
        for( int j = 0; j < n_samples; j++)
            memcpy( activations[i+1] + j*layer_ptr->n_output, layer_ptr->bias, size );
    }

    static activation_func softmax = NULL; /* Keep it static such that get_() is called only once! */
    if( !softmax )
        softmax = get_activation_func( "softmax" ); /* Slow? */

    /* Then we do the forward calculation */
    for( int i = 0; i < nn->n_layers; i++){
        const layer_t *layer_ptr = nn->layer + i;
        /* Matrix multiplication */
        cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_samples, layer_ptr->n_output, layer_ptr->n_input,
                1.0f,                /* alpha (7) */
                activations[i],      /* A     (8) */
                layer_ptr->n_input,  /* lda   (9) */
                layer_ptr->weight,   /* B     (8) */
                layer_ptr->n_output, /* ldb  (11) */
                1.0f,                /* beta (12) */
                activations[i+1],    /* C    (13) */
                layer_ptr->n_output  /* ldc  (14) */
                );
        /* Activation */
        /* ( I really hope the silly if-condition doesn't kill performance. */
        if ( layer_ptr->activation_func == softmax ){
            float *out = activations[i+1];
            for ( int j = 0; j < n_samples; j++, out += layer_ptr->n_output)
                layer_ptr->activation_func ( layer_ptr->n_output, out );
        } else {
            layer_ptr->activation_func ( layer_ptr->n_output * n_samples, activations[i+1] );
        }
    }
}
#endif /* USE_CBLAS */
