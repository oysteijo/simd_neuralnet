/* Copyright -- Øystein Schønning-Johansen 2007-2014 */

#include "neuralnet.h"
#include "simd.h"
#include "activation.h"
#include "matrix_multiply.h"
#include "c_npy.h"

#if defined(TRAINING_FEATURES)
#include "loss.h"
#include "cblas.h"
#endif

#include <stdio.h>
#include <string.h>             /* for memcpy */
#include <stdbool.h>
#include <assert.h>

#if defined(VERBOSE) 
static void neuralnet_dump( neuralnet_t *nn )
{
    for ( int i = 0; i < nn->n_layers; i++ ) {
        printf("Layer %d %7s %4d %4d ", i,
                i == 0 ? "input" : i == nn->n_layers - 1 ? "output" : "hidden",
                nn->layer[i].n_input,
                nn->layer[i].n_output);
        printf("Activation function: %s\n", get_activation_name( nn->layer[i].activation_func ));
    }
    int n_weights = 0;
    int n_biases = 0;
    for ( int i = 0; i < nn->n_layers; i++ ) {
        n_weights += nn->layer[i].n_input * nn->layer[i].n_output;
        n_biases += nn->layer[i].n_output;
    }
    printf("Total number of parameters: %d\n", n_weights + n_biases );
}
#endif

static void _weights_memory_free( neuralnet_t *nn )
{
    for ( int i = 0; i < nn->n_layers ; i++ ){
        if (nn->layer[i].weight ) simd_free( nn->layer[i].weight);
        if (nn->layer[i].bias   ) simd_free( nn->layer[i].bias);
    }
}

static bool _weights_memory_allocate( neuralnet_t *nn )
{
    for ( int i = 0; i < nn->n_layers; i++ ){
        if( (nn->layer[i].n_input < 1) || nn->layer[i].n_output < 1){
            fprintf( stderr, "Bad neural network dimensions. Can't allocate memory\n" );
            return false;
        }
    }

    for ( int i = 0; i < nn->n_layers; i++ ){
        if (NULL == (nn->layer[i].weight = simd_malloc( nn->layer[i].n_input * nn->layer[i].n_output * sizeof( float ))))
            goto weight_alloc_error;
    }

    for ( int i = 0; i < nn->n_layers; i++ ){
        if (NULL == (nn->layer[i].bias = simd_malloc( nn->layer[i].n_output * sizeof( float ))))
            goto weight_alloc_error;
    }

    /* All good! */
    return true;

weight_alloc_error:
    fprintf( stderr, "Can't allocate memory for neural network weights\n" );
    _weights_memory_free( nn );
    return false;
}

/**
  @brief Create a new neural network based ont specifications in file.
  @param filename Filename to neural network file.
  @return Pointer to newly created neural network. Returns NULL on failure. Use neuralnet_free() to free resources.
*/
neuralnet_t *neuralnet_new( const char *filename )
{
    neuralnet_t *nn;

    if ( (nn = malloc( sizeof( neuralnet_t ))) == NULL ){
        fprintf( stderr, "Cannot allocate memory for 'neuralnet_t' type.\n");
        return NULL;
    }

    /* FIXME: This code is not production quality. */
    cmatrix_t **array;
    if( NULL != (array = c_npy_matrix_array_read( filename ))) {
        size_t len = c_npy_matrix_array_length( array );
        nn->n_layers = len / 2;
        for( int i = 0; i < nn->n_layers; i++ ){
            nn->layer[i].n_input = array[i*2]->shape[0];
            nn->layer[i].n_output = array[i*2]->shape[1];
            nn->layer[i].activation_func = get_activation_func( "sigmoid" );
        }
        if( !_weights_memory_allocate( nn )){
            fprintf(stderr, "Cannot allocate memory for neural net weights.\n");
            free( nn );
            return NULL;
        }
        for( int i = 0; i < nn->n_layers; i++ ){
            cmatrix_t *weights   = array[i*2];
            cmatrix_t *bias      = array[i*2+1];
            memcpy( nn->layer[i].weight, weights->data, weights->shape[0] * weights->shape[1] * sizeof(float));
            memcpy( nn->layer[i].bias, bias->data, bias->shape[0] * sizeof(float));
        }
        return nn;
    }

    return nn;
}

/**
  @brief Free resources of a neural net.
  @param nn The neural net to free.
*/
void neuralnet_free( neuralnet_t *nn )
{
    if( !nn ) return;
    _weights_memory_free( nn );
    free( nn );
}

#if defined(VERBOSE) 
/* This is used for debug */
static void print_vector( int n, const float *v )
{
    printf("[ ");
    for (int i = 0; i < n; i++ )
        printf("% .7f ", v[i] );
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

/**
  @brief Forward calculate the neural network 
  @param nn The neural net that will do the forward calculaton. (aka. evaluate)
  @param input Pointer for an array of input features
  @param out Pointer to an array of evaluated outputs
*/
void neuralnet_predict( const neuralnet_t *nn, const float *input, float *out )
{
    /* Stack allocating memory */
    /* FIXME: Do this once and once only! */
    /* Update: Maybe not rewrite this, since it might fuck up threading... I've not tried though */

    int n_biases = 0;
    for( int i = 0; i < nn->n_layers; i++)
        n_biases += nn->layer[i].n_output;

    float SIMD_ALIGN(workmem[ n_biases ]);
    float *activations[nn->n_layers+1];
    activations[0] = (float*) input;
    activations[1] = workmem;
    for( int i = 1; i < nn->n_layers-1; i++)
        activations[i+1] = activations[i] + nn->layer[i-1].n_output;
    
    activations[nn->n_layers] = out;

    /* forward */
    for( int i = 0; i < nn->n_layers; i++){
        const layer_t *layer_ptr = nn->layer + i;
        matrix_multiply_general( 
                layer_ptr->n_input,
                layer_ptr->n_output,
                layer_ptr->weight,
                layer_ptr->bias,
                activations[i],
                activations[i+1]);
        layer_ptr->activation_func( layer_ptr->n_output, activations[i+1] );
    }
}

#if defined(TRAINING_FEATURES)
void neuralnet_save( const neuralnet_t *nn, const char *filename )
{
    cmatrix_t *array[nn->n_layers*2 + 1]; /* bias and weight for each layer pluss a terminating NULL; */

    for (int i = 0; i < nn->n_layers ; i++ ){
        /* weight */
        array[2*i] = calloc( 1, sizeof( cmatrix_t ));
        assert( array[2*i] );
        array[2*i]->data         = (char*) nn->layer[i].weight;
        array[2*i]->shape[0]     = nn->layer[i].n_input;
        array[2*i]->shape[1]     = nn->layer[i].n_output;
        array[2*i]->ndim         = 2;
        array[2*i]->endianness   = '<' ;  /* FIXME */
        array[2*i]->typechar     = 'f' ;
        array[2*i]->elem_size    = sizeof(float) ;
        array[2*i]->fortran_order= false ;

        /* bias */
        array[2*i + 1] = calloc( 1, sizeof( cmatrix_t ));
        assert( array[2*i + 1] );
        array[2*i + 1]->data         = (char*) nn->layer[i].bias;
        array[2*i + 1]->shape[0]     = nn->layer[i].n_output;
        array[2*i + 1]->ndim         = 1;
        array[2*i + 1]->endianness   = '<' ;  /* FIXME */
        array[2*i + 1]->typechar     = 'f' ;
        array[2*i + 1]->elem_size    = sizeof(float) ;
        array[2*i + 1]->fortran_order= false ;
    }
    array[2*nn->n_layers] = NULL;
    
    int retval = c_npy_matrix_array_write( filename, array );
    if( retval != nn->n_layers*2 )
        printf("Warning: Arrays written: %d  !=  2 x n_layers     (n_layers=%d)\n", retval, nn->n_layers );

    for (int i = 0; i < 2*nn->n_layers ; i++ )
        free(array[i]);
}

/* Do this better! */
static void do_nothing( unsigned int n, const float *a, float *b)
{
} 

void neuralnet_set_loss ( neuralnet_t *nn, const char *loss_name )
{
    nn->loss = get_loss_func( loss_name );
    if(!nn->loss){
        printf("Warning: Loss function '%s' not found.\n", loss_name);
        return;
    }

    /* FIXME */
    for ( int i = 0; i < nn->n_layers; i++ ){
        layer_t *layer_ptr = nn->layer + i;
        layer_ptr->activation_derivative = get_activation_derivative( layer_ptr->activation_func );
    }    

    /* Then some cleanup */
    if( nn->loss == get_loss_func( "binary_crossentropy" ) )
        if( nn->layer[nn->n_layers-1].activation_func == get_activation_func( "sigmoid" )){ 
            nn->layer[nn->n_layers-1].activation_derivative = do_nothing;
        } else {
            printf("Warning: Using 'binary_crossentropy' loss function when output activation is not 'sigmoid'.\n");
        }

    if( nn->loss == get_loss_func( "categorical_crossentropy" ) )
        if( nn->layer[nn->n_layers-1].activation_func == get_activation_func( "softmax" )){
            nn->layer[nn->n_layers-1].activation_derivative = do_nothing;
        } else {
            printf("Warning: Using 'categorical_crossentropy' loss function when output activation is not 'softmax'.\n");
        }
}

void neuralnet_backpropagation( const neuralnet_t *nn, const float *input, const float *target, float *grad )
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
        matrix_multiply_general( 
                layer_ptr->n_input,
                layer_ptr->n_output,
                layer_ptr->weight,
                layer_ptr->bias,
                activations[i],
                activations[i+1]);
        layer_ptr->activation_func( layer_ptr->n_output, activations[i+1] );
    }

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

    /* This calls the derivtive of the loss function. See loss.[ch]. */
    float *output = activations[nn->n_layers];
    assert( nn->loss );
    nn->loss( nn->layer[nn->n_layers-1].n_output, output, target, grad_b[nn->n_layers-1] );
    
    for( int layer = nn->n_layers-1; layer >= 0; layer-- ){
        const int n_inp = nn->layer[layer].n_input;
        const int n_out = nn->layer[layer].n_output;
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
        }
        /* FIXME: This will be a pointer to a funk */
        nn->layer[layer].activation_derivative( n_out, activations[layer+1], grad_b[layer] );
        
        /* This is actually the outer product */
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
    }

#if 0
    /* Update (This will be on the outside) */
    ptr = grad;
    for ( int i = 0 ; i < nn->n_layers; i++ ){
        float *b = nn->layer[i].bias;
        float *w = nn->layer[i].weight;

        for ( int j = 0, j < nn->layer[i].n_output; j++)
           *b++ += lr * *ptr++; 

        for ( int j = 0, j < nn->layer[i].n_output * nn->layer[i].n_input; j++)
           *w++ += lr * *ptr++;
    }
#endif 
}

#endif /* TRAINING_FEATURES */
