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

#if 0
/* This is used for debug */
static void print_vector( int n, const float *v )
{
    printf("[ ");
    for (int i = 0; i < n; i++ )
        printf("% .5f ", v[i] );
    printf("]\n");
}
static void print_matrix( int m, int n, const float *v )
{
    const float *ptr = v;
    printf("[\n");
    for ( int i = 0; i < m; i++ ){
        printf(" ");
        print_vector( n, ptr );
        ptr += n;
    }
    printf("]\n");
}
#endif

void neuralnet_predict_batch( const neuralnet_t *nn, const int n_samples, const float *inputs, float *output )
{
    /* Make some work memory on stack. */
    int workmem_sz = 0;
    for( int i = 0; i < nn->n_layers; i++)
        workmem_sz += nn->layer[i].n_output * n_samples;

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

#if 0
    /*  Test first mult */
    float *A = inputs;               // 2 x 6
    float *B = nn->layer[0].weight;  // 6 x 4
    float *C = activations[1];       // 2 x 4

    cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
            2, 4, 6,
            1.0f,
            A,
            6,
            B,
            4,
            1.0f,
            C,
            4
            );

    print_matrix( 2, 4, C );
#endif
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

#if 0
int main()
{
    neuralnet_t *nn = neuralnet_create( 2, INT_ARRAY( 6, 4, 2 ), STR_ARRAY( "tanh", "sigmoid" ));
    assert( nn );
    neuralnet_set_loss( nn, "binary_crossentropy"); /* I actually don't need this */
    neuralnet_initialize( nn, STR_ARRAY("xavier", "xavier" ));

    /*  Two samples */
    float SIMD_ALIGN(inputs[]) = {
        0.3f, 0.5f, 0.2f, 0.2f, 0.0f, 0.8f,
        0.5f, 0.2f, 0.2f, 0.0f, 0.8f, 0.3f
    };
    
    float SIMD_ALIGN(output[2*2]) = { 0 };

    neuralnet_save( nn, "simple_6_4_2.npz" );

    for ( int i = 0; i < 2; i++ )
        neuralnet_predict( nn, inputs + i*6, output + i*2 );

    for ( int i = 0; i < 2; i++ )
        printf("Output %d: %5.5f %5.5f\n", i, output[i*2], output[(i*2)+1]  );

    neuralnet_predict_batch( nn, 2, inputs, output);

    for ( int i = 0; i < 2; i++ )
        printf("Output %d: %5.5f %5.5f\n", i, output[i*2], output[(i*2)+1]  );

    return 0;

}
#endif
