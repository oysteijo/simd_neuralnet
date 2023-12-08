/* neuralnet_predict_batch.c - Øystein Schønning-Johansen 2023 */
/* 
 vim: ts=4 sw=4 softtabstop=4 expandtab 
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cblas.h>
#include <assert.h>
#include "neuralnet.h"
#include "simd.h"

#if RESOLVE_WITH_WORKMEMORY
/* Something along these lines. Maybe double size each time? And how do I clean up? */
static float * _get_workmemory( const size_t n_needed_floats )
{
    static size_t n_allocated_floats = 0;
    static float *mem = NULL;
    if ( n_needed_floats <= n_allocated_floats ){
        return mem;
    }
    mem = realloc( mem, n_needed_floats * sizeof(float));
    n_allocated_floats = n_needed_floats;
    return mem;
}
#endif

#ifndef STACK_LIMIT
#define STACK_LIMIT 512 * 1024
#endif
void neuralnet_predict_batch( const neuralnet_t *nn, const int n_samples, const float *inputs, float *output )
{
    /* Make some work memory on stack. */
    int workmem_sz = 0;
    for( int i = 0; i < nn->n_layers; i++)
        workmem_sz += nn->layer[i].n_output;
    workmem_sz *= n_samples;

    /* Let's see how often this this fails. */
    const size_t stack_limit = (STACK_LIMIT) / sizeof(float);
    assert( stack_limit >= workmem_sz && "Stack size limit reached - "
            "either recompile with a higher limit find another way to handle work memory" );

#if RESOLVE_WITH_RECURSION
    if( stack_limit < workmem_sz ){
        int half = n_samples >> 1;
        const int n_inputs = nn->layer[0].n_input;
        const int n_output = nn->layer[nn->n_layers-1].n_output;
        fprintf(stderr, "Warning: Stack limit reached with %d samples - recursing.\n", n_samples);
        neuralnet_predict_batch( nn, half, inputs, output );
        neuralnet_predict_batch( nn, n_samples - half, inputs+(half*n_inputs), output+(half*n_output) );
        return;    
    }
#endif
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
     * */

    float workmem[ workmem_sz ]; /* can we blow the stack here? */
    float *activations[nn->n_layers+1];
    activations[0] = (float*) inputs;
    activations[1] = workmem;

    /* print_matrix( 2, 6, inputs); */

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

    /* Then we do the forward calculation */
    for( int i = 0; i < nn->n_layers; i++){
        const layer_t *layer_ptr = nn->layer + i;
        cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_samples, layer_ptr->n_output, layer_ptr->n_input,
                1.0f,                /* alpha (7) */
                activations[i],      /* A     (8) */
                layer_ptr->n_input,  /* lda   (9) */
                layer_ptr->weight,   /* B     (8) */
                layer_ptr->n_output, /* ldb  (11) Something wrong here? */
                1.0f,                /* beta (12) */
                activations[i+1],    /* C    (13) */
                layer_ptr->n_output  /* ldc  (14) */
        );
        /* FIXME: we need a special treatment of softmax */
        layer_ptr->activation_func( layer_ptr->n_output * n_samples, activations[i+1] );
    }
}

