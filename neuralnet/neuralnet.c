/* Copyright -- Øystein Schønning-Johansen 2007-2014 */

#include "neuralnet.h"
#include "simd.h"
#include "activation.h"
#include "matrix_multiply.h"
#include "c_npy.h"

#ifndef PREDICTION_ONLY
#include "loss.h"
#endif

#include <stdio.h>
#include <string.h>             /* for memcpy */
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>
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
    printf("Total number of parameters: %u\n", neuralnet_total_n_parameters( nn ));
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
neuralnet_t *neuralnet_new( const char *filename, char *activation_funcs[] )
{
    neuralnet_t *nn;

    if ( (nn = malloc( sizeof( neuralnet_t ))) == NULL ){
        fprintf( stderr, "Cannot allocate memory for 'neuralnet_t' type.\n");
        return NULL;
    }

    /* FIXME: This code is still not production quality. */
    cmatrix_t **array;
    if( NULL != (array = c_npy_matrix_array_read( filename ))) {
        size_t len = c_npy_matrix_array_length( array );
        nn->n_layers = len / 2;
        for( int i = 0; i < nn->n_layers; i++ ){
            nn->layer[i].n_input = array[i*2]->shape[0];
            nn->layer[i].n_output = array[i*2]->shape[1];
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
            /* FIXME in far future: If the matrices are fortran order, reorganize them. Hmmm ... maybe
             * such feature belong in c_npy? */ 
            assert( weights->fortran_order == false );
            assert( bias->fortran_order == false );
        }
        
        /* Oh! What a memory leak!? */
        c_npy_matrix_array_free( array ); 

        /* Set the activation functions */
        if ( activation_funcs == NULL ) /* NULL passed into function */
            for( int i = 0; i < nn->n_layers; i++ )
                nn->layer[i].activation_func = get_activation_func("linear");

        for( int i = 0; i < nn->n_layers; i++ ){
            const char *func_name = activation_funcs[i];
            nn->layer[i].activation_func = get_activation_func( func_name ? func_name : "linear" );
        }

        return nn;
    }

    fprintf(stderr, "Neural network created, but no sizes nor activation functions were set.");
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
 
  This function only takes ONE single input sample and hence returns only one prediction.

  @param nn The neural net that will do the forward calculaton. (aka. evaluate)
  @param input Pointer for an array of input features
  @param out Pointer to an array of predictions (outputs).

  As said above, this function only take one sample, and one sample only. If you have a matrix
  of multiple samples in each row, look at the code in `evaluate.c`.
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
        vector_matrix_multiply( 
                layer_ptr->n_input,
                layer_ptr->n_output,
                layer_ptr->weight,
                layer_ptr->bias,
                activations[i],
                activations[i+1]);
        layer_ptr->activation_func( layer_ptr->n_output, activations[i+1] );
    }
}

#ifndef PREDICTION_ONLY
#define _MAX_FILENAME_LEN 128
/**
  @brief: Saves a neural network to the specified filename.

  @param nn pointer to a `neuralnet_t` structure.
  @param filename The filename where to store the neural network. The filename can contain string formating
     symbols in `printf` style.

  Example:
  \code{.c}
  neuralnet_save( nn, "after-%d-epochs.npz", epoch_count );
  \endcode

  Please note that the function do return, and any error will just output a warning in stderr. 
 */
void neuralnet_save( const neuralnet_t *nn, const char *filename, ... )
{
    if( !nn ){
        fprintf( stderr, "Warning: Cannot save neural network. No neuralnet given.\n" );
        return;
    }

    if( !filename ){
        fprintf( stderr, "Warning: Cannot save neural network. No filename given.\n" );
        return;
    }

    /* First get the full filename */
    va_list ap1, ap2;
    int len;

    va_start( ap1, filename );
    va_copy( ap2, ap1 );
    len = vsnprintf( NULL, 0, filename, ap1 );
    va_end( ap1 );

    if( len > _MAX_FILENAME_LEN ){
        /* If someone is trying to cook up a special filename that can contain executable code, I think it is wise
           to limit the length of the filename */
        fprintf( stderr, "Warning: Cannot save neural network. No filename too long."
                " Please limit the filename to %d characters.\n", _MAX_FILENAME_LEN );
        return;
    }


    char real_filename[len+1];
    vsprintf( real_filename, filename, ap2 );
    va_end( ap2 );

    /* And then the rest is the same... */

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
    
    int retval = c_npy_matrix_array_write( real_filename, array );
    if( retval != nn->n_layers*2 )
        printf("Warning: Arrays written: %d  !=  2 x n_layers     (n_layers=%d)\n", retval, nn->n_layers );

    for (int i = 0; i < 2*nn->n_layers ; i++ )
        free(array[i]);
}

/**
  @brief: Set the loss function of neural network.
  @param nn pointer to a `neuralnet_t` structure.
  @param loss_name name of the desired loss function

  This function sets the loss function for the neural network which is necessary for the backpropagation.
  Basically, you cannot do any training on the neural network without setting this. However you can do predictions
  on a neural network without it.

  In addition to setting the loss functions, this function will also set the proper derivative function
  which are used in the backpropagation algorithm.

  This function will return, but it will print warnings to stderr if it finds something is strangely set up.
 */


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
    activation_derivative do_nothing = get_activation_derivative( get_activation_func( "linear" ));
    if( nn->loss == get_loss_func( "binary_crossentropy" ) ){
        if( nn->layer[nn->n_layers-1].activation_func == get_activation_func( "sigmoid" )){ 
            nn->layer[nn->n_layers-1].activation_derivative = do_nothing;
        } else {
            printf("Warning: Using 'binary_crossentropy' loss function when output activation is not 'sigmoid'.\n");
        }
    }

    if( nn->loss == get_loss_func( "categorical_crossentropy" ) ){
        if( nn->layer[nn->n_layers-1].activation_func == get_activation_func( "softmax" )){
            nn->layer[nn->n_layers-1].activation_derivative = do_nothing;
        } else {
            printf("Warning: Using 'categorical_crossentropy' loss function when output activation is not 'softmax'.\n");
        }
    }

    if( nn->layer[nn->n_layers-1].activation_func == get_activation_func( "softmax" )){
        if( nn->loss == get_loss_func( "categorical_crossentropy" ) ){
            /* All ok. This should have been handled by the statements above */
        } else {
            printf("Warning: Using 'softmax' output activation when loss function is not 'categorical_crossentropy'.\n");
        }
    }
}

/**
  @brief: Calculates the gradient of the loss w.r.t all parameters in the neural network.

  @param nn Pointer to a `neuralnet_t` structure 
  @param input Pointer the the input vector (one sample)
  @param target Pointer to the desired target values. (of the same sample as in input)
  @param grad Pointer to the resulting gradient

  Please note that `grad` is a pointer to **all** parameters of the neural network, following after each other.
  It comes in order bias followed by weight from input to output direction.
 */
  
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
        vector_matrix_multiply( 
                layer_ptr->n_input,
                layer_ptr->n_output,
                layer_ptr->weight,
                layer_ptr->bias,
                activations[i],
                activations[i+1]);
        layer_ptr->activation_func( layer_ptr->n_output, activations[i+1] );
    }

    /* backward */

    /* First we set the grad vector to 0.0. The caller always seems forgets! */
    unsigned int n_param = neuralnet_total_n_parameters( nn );
    memset( grad, 0, n_param * sizeof(float));

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
            matrix_vector_multiply(
                    nn->layer[layer+1].n_input,   /* n */
                    nn->layer[layer+1].n_output,  /* m */
                    nn->layer[layer+1].weight,    /* matrix */
                    grad_b[layer+1],              /* v (The vector) */
                    grad_b[layer] );              /* The result (y) */
        }
        nn->layer[layer].activation_derivative( n_out, activations[layer+1], grad_b[layer] );
        
        /* This is actually the outer product */
        vector_vector_outer( n_inp, n_out, activations[layer], grad_b[layer], grad_w[layer] );
    }
}

/**
  @brief: A function to update all parameters in the neural network. This function is typically called from an optimizer.

  @param nn Pointer to `neuralnet_t` structure.
  @param delta_w Pointer to the delta of the parameters

  Basically this function does:  params += delta_w
 */
void neuralnet_update( neuralnet_t *nn, const float *delta_w )
{ 
    const float *ptr = delta_w;
    for ( int l = 0; l < nn->n_layers; l++ ){
        const int n_inp = nn->layer[l].n_input;
        const int n_out = nn->layer[l].n_output;
        /* Update the biases */
        vector_accumulate_unaligned( n_out, nn->layer[l].bias, ptr );
        ptr += n_out;
        /* Update the weights */
        vector_accumulate_unaligned( n_out * n_inp, nn->layer[l].weight, ptr );
        ptr += n_inp * n_out;
    }
}

/**
  @brief: Creates a new neural network structure and allocates memory for the parameters.

  @param n_layers The desired number of layers in the neural network to be created. Note that the counting is modern,
  such that the classic input->hidden->output structure which in the 90s where called a three layer MLP is
  actually 2 layers in this system.
  @param sizes An integer array if size n_layers+1. The int values indicate the input and output sizes of the layers in neural network.
  @param activation_funcs An array of lenght n_layers with strings describing the activation function to be used..

  returns A newly created `neuralnet_t` structure, or NULL with failure.

  There are two conveinient macros defined to hide compound literal if the arrays are not created
  INT_ARRAY and STR_ARRAY,

  Example:
  A (classic) neural network with three inputs, four hidden nodes and 2 outputs can be defined like this:
  \code{.c}
  neuralnet_t *nn = neuralnet_create( 2,                           // n_layers
                                    INT_ARRAY( 3, 4, 2 ),          // sizes
                                    STR_ARRAY( "tanh", "sigmoid" ) // activation functions
                                    );
  \endcode

  To initialize proper random parameter values, use `neuralnet_initialize()`.
*/
neuralnet_t * neuralnet_create( const int n_layers, int sizes[], char *activation_funcs[] )
{
    if( n_layers < 1 ){
        fprintf( stderr, "You need at least one layer in your neural network. %d layers does not make sense.\n", n_layers );
        return NULL;
    } 
    /* FIXME: I think we can do better. There should not be a hard limit. */
    if( n_layers >  NN_MAX_LAYERS ){
        fprintf( stderr, "Oh... you've hit a limit in the system -- please recompile and set NN_MAX_LAYERS to a higher value.");
        return NULL;
    } 

    /* Sanity check on sizes */
    for( int i = 0; i < n_layers; i++ )
        if( sizes[i] < 1 ){
            fprintf(stderr, "Input size in layer %d is %d. That does not make sense!\n", i, sizes[i]);
            return NULL;
        }
    if( sizes[n_layers] < 1 ){
        fprintf(stderr, "Output size of neural net is %d. That does not make sense!\n", sizes[n_layers]);
        return NULL;
    }

    neuralnet_t *nn;
    if ( (nn = malloc( sizeof( neuralnet_t ))) == NULL ){
        fprintf( stderr, "Cannot allocate memory for 'neuralnet_t' type.\n");
        return NULL;
    }

    nn->n_layers = n_layers;
    for( int i = 0; i < nn->n_layers; i++ ){
        nn->layer[i].n_input  = sizes[i];
        nn->layer[i].n_output = sizes[i+1];
    }

    if( !_weights_memory_allocate( nn )){
        fprintf(stderr, "Cannot allocate memory for neural net weights.\n");
        free( nn );
        return NULL;
    }

    /* Set the activation functions */
    for( int i = 0; i < nn->n_layers; i++ ){
        const char *func_name = activation_funcs[i];
        nn->layer[i].activation_func = get_activation_func( func_name ? func_name : "linear" );
    }

    return nn;
}

/* functions for initializing weights of fresh neural networks */

/* Returns a random float number from -1 to +1 with a uniform probability distribution. */
static float random_uniform()
{
    return (float) 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
}

static float random_normal()
{
    static float v, fac;
    static bool use_prev = false;
    float S, Z, u;
    if(use_prev)
        Z = v * fac;
    else {
        do {
            u = random_uniform();  /* from -1 to +1 */
            v = random_uniform();  /* from -1 to +1 */

            S = u*u + v*v;
        } while (S >= 1.0f);

        fac = sqrtf( -2.0f * logf(S) / S );
        Z = u * fac;
    }

    use_prev = !use_prev;
    return Z;
}

static void fill_data( unsigned int n, float (*dist)(), float scale, float *data )
{
    float *ptr = data;
    for ( unsigned int i = n; i--; )
        *ptr++ = dist() * scale;
}

#define foreach_str( iter, ... ) \
    for( char **iter = (char*[]){__VA_ARGS__, NULL}; *iter; iter++ )
#define streq(a,b) (strcmp((a),(b)) == 0)

void neuralnet_initialize( neuralnet_t *nn, ... )
{
    assert( nn );

    va_list argp;
    va_start( argp, nn );

    for ( int i = 0; i < nn->n_layers ; i++ ){
        const int n_inp = nn->layer[i].n_input;
        const int n_out = nn->layer[i].n_output;

        char *initializer = va_arg( argp, char* );

        if( streq( initializer, "xavier" ))
            fill_data( n_inp * n_out, random_uniform, sqrtf(6.0f / (n_inp+n_out)), nn->layer[i].weight ); /* Xavier */
        else if( streq( initializer, "kaiming" ))
            fill_data( n_inp * n_out, random_normal, sqrtf(2.0f/n_inp), nn->layer[i].weight ); /* Kaiming */
        else {
            /* OK ... initializer was neither "kaiming" nor "xavier", let's do some guessing. */
            bool initialized = false;
            const char *activation_name = get_activation_name( nn->layer[i].activation_func );
            /* The concept is simple, if the activation is in this list -- use Xavier init */
            foreach_str( activation, "sigmoid", "tanh", "softmax", "hard_sigmoid", "softsign" ){
                if( streq( *activation, activation_name ) ){
                    fill_data( n_inp * n_out, random_uniform, sqrtf(6.0f / (n_inp+n_out)), nn->layer[i].weight ); /* Xavier */
                    initialized = true;
                    break;
                }
            }
            /* The concept is simple, if the activation is in this list -- use Kaiming init (aka He) */
            foreach_str( activation, "relu", "softplus" ){
                if( streq( *activation, activation_name ) ){
                    fill_data( n_inp * n_out, random_normal, sqrtf(2.0f/n_inp), nn->layer[i].weight ); /* Kaiming */
                    initialized = true;
                    break;
                }
            }

            if(!initialized) { /* What the f... is this ? Well... maybe linear or exponential. BTW what about RBF? */
                fprintf(stderr, "Warning: Initializers not recognized at layer %d. normal distributed random values used.\n", i);
                fill_data( n_inp * n_out, random_normal, 1.0f, nn->layer[i].weight );
            }
        }
        /* Fill bias terms with zeros. OK? */
        memset( nn->layer[i].bias, 0, n_out * sizeof(float));
    }
    va_end( argp );
}

#endif /* PREDICTION_ONLY */
