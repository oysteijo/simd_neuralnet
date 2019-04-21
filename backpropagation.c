#include "neuralnet.h"
#include "simd.h"
#include "matrix_multiply.h"
#include <string.h>
#include <cblas.h>

void sigmoid_diff( int n, const float *act, float *v )
{
    for( int i=0; i < n; i++ )
        v[i] *= act[i]*(1.0f-act[i]);
}

#if 0
static void print_vector( int n, const float *v )
{
    printf("[ ");
    for (int i = 0; i < n; i++ )
        printf("% .4e ", v[i] );
    printf("]\n");
}
#endif 

void backpropagation( const neuralnet_t *nn, const float *input, const float *target, float *grad )
{
    int n_biases = 0;
    for( int i = 0; i < nn->n_layers; i++)
        n_biases += nn->layer[i].n_output;

    float SIMD_ALIGN(workmem[ n_biases + nn->layer[0].n_input ]);
    float *activations[nn->n_layers+1];
    activations[0] = (float*) input;
    activations[1] = workmem;
    for( int i = 1; i < nn->n_layers; i++)
        activations[i+1] = activations[i] + nn->layer[i-1].n_output;
    
    /* forward */
    for( int i = 0; i < nn->n_layers; i++){
        const layer_t *layer_ptr = nn->layer + i;
        // printf("Layer: %d\n inputs: %2d\noutputs: %2d\n", i, layer_ptr->n_input, layer_ptr->n_output); 
        matrix_multiply_general( 
                layer_ptr->n_input,
                layer_ptr->n_output,
                layer_ptr->weight,
                layer_ptr->bias,
                activations[i],
                activations[i+1]);
        // print_vector( layer_ptr->n_output, y[i] );
        layer_ptr->activation_func( layer_ptr->n_output, activations[i+1] );
        // print_vector( layer_ptr->n_output, activations[i+1] );
    }
    // print_vector( nn->layer[nn->n_layers-1].n_output, activations[nn->n_layers] );


    /* backward */

    /* Set up some pointers */
    float *grad_b[nn->n_layers];
    float *grad_w[nn->n_layers];
    float *ptr = grad;
    for( int i = 0; i < nn->n_layers; i++ ) {
        const int n_inp = nn->layer[i].n_input;
        const int n_out = nn->layer[i].n_output;
        grad_b[i] = ptr;
        ptr += n_out;
        grad_w[i] = ptr;
        ptr += n_inp*n_out;
    }
    /* This loop is the derivtive of the root mean squared error. This has to be changed
     * for other loss functions, and also vectorized to SIMD instructions. I hence think that
     * this can be abstracted away with a pointer to a function */ 
    float *output = activations[nn->n_layers];
    const int n_out = nn->layer[nn->n_layers-1].n_output;
    for ( int i = 0; i < n_out; i++ ){
        grad_b[nn->n_layers-1][i]  = 2.0f * ( output[i] - target[i] ) / (float) n_out;
    }

    // for l in range(1, len(self.layers)+1):
    for( int layer = nn->n_layers-1; layer >= 0; layer-- ){
        const int n_inp = nn->layer[layer].n_input;
        const int n_out = nn->layer[layer].n_output;
        // printf("n_inp: %d\n", n_inp);
        // printf("n_out: %d\n", n_out);
        if( layer != nn->n_layers-1 ) {
            cblas_sgemv( CblasRowMajor, CblasNoTrans,
                    nn->layer[layer+1].n_input,
                    nn->layer[layer+1].n_output,
                    1.0f, 
                    nn->layer[layer+1].weight,
                    nn->layer[layer+1].n_output,
                    grad_b[layer+1], 1,
                    0.0f, /* beta */
                    grad_b[layer], 1 );
            // print_vector( n_out, grad_b[layer]);
        }
        /* FIXME: This will be a pointer to a funk */
        sigmoid_diff( n_out, activations[layer+1], grad_b[layer] );
        // print_vector( n_out, grad_b[layer] );
        
        cblas_sger(CblasRowMajor, /* you’re using row-major storage */
           n_inp,                 /* the matrix X has dx1 rows ...  */
           n_out,                 /*  ... and dx2 columns.          */
           1.0,                   /* scale factor to apply to x1x2' */
           activations[layer], 
           1,                     /* stride between elements of x1. */
           grad_b[layer],
           1,                     /* stride between elements of x2. */
           grad_w[layer],
           n_out);                /* leading dimension of matrix X. */
        // print_vector( n_out * n_inp , grad_w[layer]);
    }
}
