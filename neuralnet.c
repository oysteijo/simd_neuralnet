/* Copyright -- Øystein Schønning-Johansen 2007-2014 */

#include "neuralnet.h"
#include "simd.h"
#include "activation.h"
#include "matrix_multiply.h"

#include <stdio.h>
#include <string.h>             /* for memcpy */
#include <stdbool.h>
#include <stdlib.h>             /* for drand48 */
#include <time.h>               /* for time */
#include <ctype.h>              /* for isspace */
#include <math.h>               /* for sqrtf */
#include <assert.h>

enum { READ_FROM_FILE = 0, RANDOM_WEIGHTS };

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

#if defined(TRAINING_FEATURES)
void dummy_trainer( neuralnet_t *nn, const float * input, const float *desired, const void *unused );
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

static bool str_is_empty(const char *s) {
    while (*s != '\0') {
        if (!isspace((unsigned char)*s))
        return false;
        s++;
    }
    return true;
}

#define LINE_MAX 1024

static bool _read_header_info(neuralnet_t *nn, const char *filename, int *type)
{
    FILE *fp; 
    if ( NULL == (fp = fopen( filename, "r"))){
        perror( filename );
        return false;
    }

    /* ASCII file */
    char line[LINE_MAX];
    int layer_size[MLP_MAX_LAYERS + 1 ] = { 0 };

    do {
        fgets( line, LINE_MAX, fp ); /* Read header line */
    } while( (line[0] == '#') || (str_is_empty(line)));

    char *delimiters = " \t\n;:,";
    char *token = strtok( line, delimiters);
    if( token )
        layer_size[0] = strtoll( token, NULL, 10 );
    else {
        fprintf( stderr, "Cannot read neuralnet topology string\n");
    return false;
    }

    for( int i = 1; i <= MLP_MAX_LAYERS ; i++ ){
        token = strtok( NULL, delimiters);
        char *end = token;
        if( token ) layer_size[i] = strtoll( token, &end, 10 );
        nn->n_layers = i;
        if( end == token ) break;
    }

    /* There is one less layer than sizes */
    nn->n_layers--;

    for ( int i = 0; i < nn->n_layers; i++ ){
        nn->layer[i].n_input = layer_size[i];
        nn->layer[i].n_output = layer_size[i+1];
    }

    /* read activation functions */
    if ( token ) {
        if( NULL == (nn->layer[0].activation_func = get_activation_func( token ))){
            fprintf( stderr, "Activation function at input layer not set.\n");
            fprintf( stderr, "Probed name was '%s'.\n", token );
            goto error;
        }
    } else {
        fprintf( stderr, "Activation function name for input layer not found.\n");
        goto error;
    }
    
    for ( int i = 1; i < nn->n_layers; i++ ){
        token = strtok( NULL, delimiters);
        if( token ){
            if( NULL == (nn->layer[i].activation_func = get_activation_func( token ))){
                fprintf( stderr, "Activation function at layer %d not set.\n", i);
                fprintf( stderr, "Probed name was '%s'.\n", token );
                goto error;
            }
        } else {
            fprintf( stderr, "Activation function name not found. (Layer %d)\n", i);
            goto error;
        }
    }

    /* This is silly, but let's fix this later. The point is that it should read values in
     * from the file as long as there are values. Then, when no more values... pick random values. */
    if ( fgets( line, LINE_MAX, fp ) == NULL){
        fprintf( stderr, "Cannot read initialisation line of '%s'\n", filename );
        goto error;
    }

    if (strncmp( "random", line, strlen("random")) == 0 )
        *type = RANDOM_WEIGHTS;
    else
        *type = READ_FROM_FILE;

    fclose(fp);
#if defined(VERBOSE)
    neuralnet_dump( nn );
#endif
    return true;
error:
    fclose( fp );
    return false;
}

static bool _initialize_weights_from_file (neuralnet_t *nn, const char *filename)
{
    FILE *fp; 

    /* Open file */
    if ( NULL == (fp = fopen( filename, "r"))){
        perror( filename );
        return false;
    }

    char line[LINE_MAX]; 

    /* Skip comments and header */
    do {
        fgets( line, LINE_MAX, fp ); /* Read header line */
    } while( (line[0] == '#') || (str_is_empty(line)));

    /* FIXME: This is the first step in the direction of generalizing the weight pointers.
     * Maybe this can be developed further? General number of layers, etc. */

    /* Read weights */
    for( int j = 0; j < nn->n_layers; j++ ){         
        float *pr = nn->layer[j].weight;
        int size = nn->layer[j].n_input * nn->layer[j].n_output;
        for (int i = size; i; i--){
            if( fgets( line, LINE_MAX, fp) == NULL ){
                fprintf( stderr, "Can't read more lines from file '%s'.\n", filename );
                goto error;
            }
            char *endptr;
            *pr++ = strtof( line, &endptr );
            if( line == endptr ){
                fprintf( stderr, "Can't read value '%s' from file '%s'.\n", line, filename );
                goto error;
            }
        }
    }

    /* Read bias */  /* and DRY!!! FIXME */
    for( int j = 0; j < nn->n_layers; j++ ){
        float *pr = nn->layer[j].bias;
        int size = nn->layer[j].n_output;
        for (int i = size; i; i--){
            if( fgets( line, LINE_MAX, fp) == NULL ){
                fprintf( stderr, "Can't read more lines from file '%s'.\n", filename );
                goto error;
            }
            char *endptr;
            *pr++ = strtof( line, &endptr );
            if( line == endptr ){
                fprintf( stderr, "Can't read value '%s' from file '%s'.\n", line, filename );
                goto error;
            }
        }
    }

    fclose( fp );

#if defined(VERBOSE)
    fprintf( stdout, "Initialized neural net from file: %s\n", filename ); 
#endif
    return true;
error:
    fclose( fp );
    return false;
}

#if 0
static void _initialize_weights_random( neuralnet_t *nn )
{
    struct {
        int size;
        float *p_start;
    } loopdata[4] = { 
        { .size = nn->n_input * nn->n_hidden,  .p_start = nn->weight_h },
        { .size = nn->n_hidden, .p_start = nn->bias_h },
        { .size = nn->n_hidden * nn->n_output, .p_start = nn->weight_o },
        { .size = nn->n_output, .p_start = nn->bias_o } 
    };

#if defined( _SVID_SOURCE) || defined( _XOPEN_SOURCE )
    srand48( time( NULL ) );
#else
    srand( time( NULL ) );
#endif
    for( int j = 0; j < 4; j++ ){
        float *pr = loopdata[ j ].p_start;
        for (int i = loopdata[j].size; i; i--)
#if defined( _SVID_SOURCE) || defined( _XOPEN_SOURCE )
            *pr++ = (float) ( 2.0 * drand48() - 1.0);
#else
            *pr++ = (float) (2.0f*rand()/(float)(RAND_MAX) - 1.0f);
#endif
    }

    /* Normalize hidden parameters */
    float vec_len = 0.0f;
    for ( int j = 0 ; j < 2; j++ ){
        float *pr = loopdata[ j ].p_start;
        for (int i = loopdata[j].size; i; i--, pr++)
            vec_len += *pr * *pr;
    }
    vec_len = sqrtf( vec_len );
    for ( int j = 0 ; j < 2; j++ ){
        float *pr = loopdata[ j ].p_start;
        for (int i = loopdata[j].size; i; i--, pr++)
            *pr /= vec_len;
    }

    /* Normalize output parameters */
    vec_len = 0.0f;
    for ( int j = 2 ; j < 4; j++ ){
        float *pr = loopdata[ j ].p_start;
        for (int i = loopdata[j].size; i; i--, pr++)
            vec_len += *pr * *pr;
    }
    vec_len = sqrtf( vec_len );
    for ( int j = 2 ; j < 4; j++ ){
        float *pr = loopdata[ j ].p_start;
        for (int i = loopdata[j].size; i; i--, pr++)
            *pr /= vec_len;
    }

    return;
}
#endif

/**
  @brief Create a new neural network based ont specifications in file.
  @param filename Filename to neural network file.
  @return Pointer to newly created neural network. Returns NULL on failure. Use neuralnet_free() to free resources.
*/
#include "c_npy.h"
neuralnet_t *neuralnet_new( const char *filename )
{
    neuralnet_t *nn;
    int inittype;

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


    if ( !_read_header_info( nn, filename, &inittype )){
        fprintf(stderr, "Cannot read header info from file: '%s'\n", filename );
        free(nn);
        return NULL;
    }

    if( !_weights_memory_allocate( nn )){
        fprintf(stderr, "Cannot allocate memory for neural net weights.\n");
        free( nn );
        return NULL;
    }

    if ( (inittype == READ_FROM_FILE) && !_initialize_weights_from_file( nn, filename )) { 
        fprintf(stderr, "Cannot create neural network. Filename provided: '%s'\n", filename);
        neuralnet_free( nn );
        return NULL;
    }

#if 0
    if ( inittype == RANDOM_WEIGHTS )
        _initialize_weights_random( nn );
#endif

#if defined(TRAINING_FEATURES)
    neuralnet_set_trainer( nn, dummy_trainer );
#endif
    return nn;
}

#ifdef READ_NUMPY_NEURAL_NETWORKS
#include "c_npy.h"
/**
  @brief Create a new neural network based on a numpy `.npz` file.
  @param filename Filename of the fail containing the neural network.
  @return Pointer to newly created neural network. Returns NULL on failure. Use neuralnet_free() to free resources.

  The format is simple. Each matrix of the neural network is stored in a numpy `.npz` file in the input to output direction.
  Each layer is then stored as weight matrix, bias matrix and a byte/char matrix that makes a string representation of a
  activation function.

  A classic MLP neural network with 32 inputs 16 hidden units and 4 outpus, will hence be represented as:

  Layer 1
  numpy narray of shape (32,16) and np.float32 type
  numpy narray of shape (16,)   and np.float32 type.
  numpy narray of bytes where the bytes makes a string that describes the activation function used for that layer.

  Layer 2
  numpy narray of shape (16,4) and np.float32 type
  numpy narray of shape (4,)   and np.float32 type.
  numpy narray of bytes where the bytes makes a string that describes the activation function used for that layer.

  An example code of how to generate a neural network from Python will be provided.
*/

neuralnet_t *neuralnet_new_from_numpy( const char *filename )
{
    neuralnet_t *nn;
    int inittype;

    if ( (nn = malloc( sizeof( neuralnet_t ))) == NULL ){
        fprintf( stderr, "Cannot allocate memory for 'neuralnet_t' type.\n");
        return NULL;
    }

    cmatrix_t **arr = c_npy_matrix_array_read  ( filename );
    if( !arr ) {
        fprintf( stderr, "Cannot read numpy file '%s'.\n", filename);
        neuralnet_free( nn );
        return NULL;
    }

    size_t n_arrays = c_npy_matrix_array_length( (const cmatrix_t **) arr );

    if( (n_arrays % 3) != 0 ){
        fprintf( stderr, "Number of arrays in files cannot form neural network ('%s').\n", filename);
        neuralnet_free( nn );
        c_npy_matrix_array_free( arr );
        return NULL;
    }

    nn->n_layers = n_arrays / 3;
    
    /* Now there are many options. I can read the numpy arrays and memcpy the data into the layers.
       or I can use mmap(), or I can just use the data as they are read. I think I actually need to
       align for SMID. I think I read and copy for this reason. */
#if 0
typedef struct _cmatrix_t {
    char    *data;
    size_t   shape[ C_NPY_MAX_DIMENSIONS ];
    int32_t  ndim;
    char     endianness;
    char     typechar;
    size_t   elem_size;
    bool     fortran_order;
} cmatrix_t;
#endif


    for( int i = 0; i < nn->n_layers; i++ ){
        cmatrix_t *weights   = arr[i*3];

        assert( weights->ndim == 2 );  /* There is no support for Convolutional nets etc. yet! */
        nn->layer[i].n_input  = weights->shape[0]; 
        nn->layer[i].n_output = weights->shape[1]; 
    }
    /* FIXME: Check sanity of matrices the n_input of one layer should match the n_output of the previous */ 


    if( !_weights_memory_allocate( nn )){
        fprintf(stderr, "Cannot allocate memory for neural net weights.\n");
        free( nn );
        return NULL;
    }


    for( int i = 0; i < nn->n_layers; i++ ){
        cmatrix_t *weights   = arr[i*3];
        cmatrix_t *bias      = arr[i*3+1];
        cmatrix_t *act_array = arr[i*3+2];
        

        assert( weights->ndim == 2 );  /* There is no support for Convolutional nets etc. yet! */
        assert( weights->elem_size == sizeof(float) );
        assert( weights->typechar  == 'f' );

        assert( bias->ndim == 1 );
        assert( bias->elem_size == sizeof(float) );
        assert( bias->typechar  == 'f' );

        assert( act_array->ndim == 1 );
        assert( act_array->elem_size == sizeof(char) );
        assert( act_array->typechar  == 'S' );

        memcpy( nn->layer[i].weight, weights->data, weights->shape[0] * weights->shape[1] * sizeof(float));
        memcpy( nn->layer[i].bias, bias->data, bias->shape[0] * sizeof(float));

        /* get_activation_func() uses strcmp() which requeres that the string is NULL terminated. Hence the copying */
        char af_name[256] = { '\0' };
        memcpy( af_name, act_array->data, act_array->shape[0] * sizeof(char));
        nn->layer[i].activation_func = get_activation_func( af_name );
    }

    c_npy_matrix_array_free( arr );
    return nn;

}
#endif



#if 0
void neuralnet_scale_parameters( const neuralnet_t *nn, int idx, float scale )
{
    struct {
        int size;
        float *p_start;
    } loopdata[4] = { 
        { .size = nn->n_input * nn->n_hidden,  .p_start = nn->weight_h },
        { .size = nn->n_hidden, .p_start = nn->bias_h },
        { .size = nn->n_hidden * nn->n_output, .p_start = nn->weight_o },
        { .size = nn->n_output, .p_start = nn->bias_o } 
    };

    if ( idx > 3 ) return;

    float *pr = loopdata[ idx ].p_start;
    for (int i = loopdata[idx].size; i; i--, pr++)
        *pr *= scale;
}
#endif
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

static void print_vector( int n, const float *v )
{
    printf("[ ");
    for (int i = 0; i < n; i++ )
        printf("% .7f ", v[i] );
    printf("]\n");
}

/**
  @brief Forward calculate the neural network 
  @param nn The neural net that will do the forward calculaton. (aka. evaluate)
  @param input Pointer for an array of input features
  @param out Pointer to an array of evaluated outputs
*/
void neuralnet_evaluate( const neuralnet_t *nn, const float *input, float *out )
{
    /* Stack allocating memory */
    /* FIXME: Do this once and once only! */
    /* Update: Maybe not rewrite this, since it might fuck up threading... I've not tried though */
    int n_biases = 0;
    for( int i = 0; i < nn->n_layers; i++)
        n_biases += nn->layer[i].n_output;

    float SIMD_ALIGN(workmem[ n_biases ]);
    float *y[nn->n_layers];
    y[0] = workmem;
    for( int i = 0; i < nn->n_layers - 1; i++)
        y[i+1] = y[i] + nn->layer[i].n_output;

    // layer_t *p_layer = nn->layer;
    print_vector( nn->layer[0].n_input, input );

    for( int i = 0; i < nn->n_layers; i++){
        const layer_t *layer_ptr = nn->layer + i;
        /* printf("Layer: %d\n inputs: %2d\noutputs: %2d\n", i, layer_ptr->n_input, layer_ptr->n_output); */
        matrix_multiply_general( 
                layer_ptr->n_input,
                layer_ptr->n_output,
                layer_ptr->weight,
                layer_ptr->bias,
                i ? y[i-1] : input,
                y[i]);
        print_vector( layer_ptr->n_output, y[i] );
        layer_ptr->activation_func( layer_ptr->n_output, y[i] );
        print_vector( layer_ptr->n_output, y[i] );
    }
    memcpy( out, y[nn->n_layers-1], 4 * sizeof(float));
    return;

    /* output layer is transposed */
    // layer_t outlayer = nn->layer[nn->n_layers];
    const layer_t *layer_ptr = nn->layer + nn->n_layers - 1;
    matrix_multiply_output( 
            layer_ptr->n_input,
            layer_ptr->n_output,
            layer_ptr->weight,
            layer_ptr->bias,
            y[nn->n_layers-2], out );
    layer_ptr->activation_func( layer_ptr->n_output, out );
}

#if defined( TRAINING_FEATURES )
void neuralnet_save( const neuralnet_t *nn, const char *filename )
{
    FILE *fp;

    if(!(fp = fopen( filename, "w" ))){
        perror( filename );
        return;
    }

    /* ASCII file */
    fprintf( fp, "%u ", nn->layer[0].n_input);
    for ( int i = 0; i < nn->n_layers; i++ )
        fprintf( fp, "%u ", nn->layer[i].n_output );

    for ( int i = 0; i < nn->n_layers; i++ )
        fprintf( fp, "%s ", get_activation_name( nn->layer[i].activation_func ));

    fprintf( fp, "\n");

    for( int j = 0; j < nn->n_layers; j++ ){
        float *pr = nn->layer[j].weight;
        int size = nn->layer[j].n_input * nn->layer[j].n_output;
        for (int i = size; i; i--)
            fprintf( fp, "%.10g\n",  *pr++ );
    }

    for( int j = 0; j < nn->n_layers; j++ ){
        float *pr = nn->layer[j].bias;
        int size = nn->layer[j].n_output;
        for (int i = size; i; i--)
            fprintf( fp, "%.10g\n",  *pr++ );
    }

    fclose( fp );
}

void neuralnet_set_trainer( neuralnet_t *nn, trainfunc tf)
{
    assert( tf );
    nn->train = tf;
}

void neuralnet_train( neuralnet_t *nn, const float * input, const float * desired, const void *data)
{
    nn->train( nn, input, desired, data );
}
#endif /* TRAINING_FEATURES */

/* FIXME: Check how to suppress warnings with other compilers */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
void dummy_trainer( neuralnet_t *nn, const float * input, const float *desired, const void *unused )
{
    fprintf( stderr, "No parameter adjustments done. Call neuralnet_set_trainer() to add a train function.\n" );
    return;
}
#if defined(__GNUC__)
#pragma GCC diagnostic push
#endif
