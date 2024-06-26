#include "npy_array.h"
#include "npy_array_list.h"
#include "neuralnet.h"
#include "simd.h"

#include "optimizer.h"
#include "optimizer_implementations.h"
#include "loss.h"

#include "callback.h"
#include "logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static uint32_t read_uint_big_endian( FILE *fp )
{
    union { unsigned char c[4]; uint32_t i; } val;
    if( fread( &val, sizeof(val), 1, fp ) != 1)
        fprintf(stderr, "Warning - could not read uint in big endian order. (line: %3d)\n", __LINE__ );

    unsigned char tmp;
    tmp = val.c[0];
    val.c[0] = val.c[3];
    val.c[3] = tmp;

    tmp = val.c[1];
    val.c[1] = val.c[2];
    val.c[2] = tmp;
    return val.i;
}

float *read_idx_features( const char *filename, uint32_t *n_samples, uint32_t *n_features )
{
    FILE *fp = fopen(filename, "rb");
    if( !fp ) return NULL;

    /* This text is copied for the documentation on Yann's website */
    /*
    The magic number is an integer (MSB first). The first 2 bytes are always 0.

        The third byte codes the type of the data:
        0x08: unsigned byte
        0x09: signed byte
        0x0B: short (2 bytes)
        0x0C: int (4 bytes)
        0x0D: float (4 bytes)
        0x0E: double (8 bytes)

        The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
    */
    unsigned char magic[ 4 ];
    if( fread( magic, sizeof(unsigned char), 4, fp ) != 4 )
        fprintf(stderr, "Warning - cannot read 4 bytes of magic. (line: %3d)\n", __LINE__ );

    assert( magic[0] == 0 );
    assert( magic[1] == 0 );
    assert( magic[2] == 0x08 );  /* We expect unsigned bytes */

    int n_dim = (int) magic[3];

    /* if this is features we expect n_dim to be three */
    assert( n_dim == 3 );

    /* Reuse the magic to get the right (little endian) integer */
    *n_samples = read_uint_big_endian( fp );

    uint32_t n_rows = read_uint_big_endian( fp );
    uint32_t n_cols = read_uint_big_endian( fp );

    *n_features = n_rows * n_cols;

    unsigned char *raw = malloc( *n_samples * *n_features * sizeof( unsigned char ));
    if( fread( raw, sizeof( unsigned char ), *n_samples * *n_features, fp ) != (size_t) *n_samples * *n_features)
        fprintf(stderr, "Warning - Cannot read the feature data. (line: %3d)\n", __LINE__ );

    float   *features = malloc( *n_samples * *n_features * sizeof( float ));

    unsigned int n = *n_samples * *n_features;
    float *fptr = features;
    unsigned char *raw_ptr = raw;

    while( n-- )
        *fptr++ = (float) *raw_ptr++ / 255.0f;

    free( raw );
    fclose( fp );
    return features;
}

float *read_idx_labels( const char *filename, uint32_t *n_samples, uint32_t n_targets )
{
    FILE *fp = fopen(filename, "rb");
    if( !fp ) return NULL;

    /* This text is copied for the documentation on Yann's website */
    /*
    The magic number is an integer (MSB first). The first 2 bytes are always 0.

        The third byte codes the type of the data:
        0x08: unsigned byte
        0x09: signed byte
        0x0B: short (2 bytes)
        0x0C: int (4 bytes)
        0x0D: float (4 bytes)
        0x0E: double (8 bytes)

        The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
    */
    unsigned char magic[ 4 ];
    if( fread( magic, sizeof(unsigned char), 4, fp ) != 4 )
        fprintf(stderr, "Warning - cannot read 4 bytes of magic. (line: %3d)\n", __LINE__ );

    assert( magic[0] == 0 );
    assert( magic[1] == 0 );
    assert( magic[2] == 0x08 );  /* We expect unsigned bytes */

    int n_dim = (int) magic[3];

    /* if this is labels we expect n_dim to be one */
    assert( n_dim == 1 );

    /* Reuse the magic to get the right (little endian) integer */
    *n_samples = read_uint_big_endian( fp );

    unsigned char *raw = malloc( *n_samples * sizeof( unsigned char ));
    if ( fread( raw, sizeof( unsigned char ), *n_samples, fp ) != *n_samples)
        fprintf(stderr, "Warning - Cannot read target values data. (line: %3d)\n", __LINE__ );

    float   *labels = calloc( *n_samples * n_targets, sizeof( float ));

    /* Do one-hot encoding of target labels */
    for( unsigned int i = 0; i < *n_samples; i++ ){
        *((labels + (i * n_targets)) + *(raw+i)) = 1.0f;   /* Hehe! Did you get that? */
    }

    free( raw );
    fclose( fp );
    return labels;
}

int main( int argc, char *argv[] )
{
    uint32_t n_train_samples;
    uint32_t n_test_samples;
    uint32_t n_input_features;
    uint32_t n_output_targets = 10;

    float *train_features = read_idx_features( "train-images-idx3-ubyte", &n_train_samples, &n_input_features );
    float *train_labels   = read_idx_labels  ( "train-labels-idx1-ubyte", &n_train_samples, n_output_targets );
    float *test_features  = read_idx_features( "t10k-images-idx3-ubyte", &n_test_samples, &n_input_features );
    float *test_labels    = read_idx_labels  ( "t10k-labels-idx1-ubyte", &n_test_samples, n_output_targets );

    if( !train_labels   || !train_features || !test_labels    || !test_features ){
        fprintf(stderr, "Cannot read datafiles. See README.\n");
        return 0;
    }

    /* Set up a new Neural Network */
    neuralnet_t *nn = neuralnet_create( 3,
            INT_ARRAY( n_input_features, 256, 128, n_output_targets ),
            STR_ARRAY( "relu", "relu", "softmax" ) );
    assert( nn );

    neuralnet_initialize( nn, NULL ); 
    neuralnet_set_loss( nn, "categorical_crossentropy" );

    /* Training with plain Stochastic Gradient Decsent (SGD) */    
    const float learning_rate = 0.001f;
#if 0
    optimizer_t *opt = optimizer_new( nn, 
            OPTIMIZER_CONFIG(
                .batchsize = 16,
                .shuffle   = true,
                .run_epoch = adamw_run_epoch,
                .settings  = ADAMW_SETTINGS( .learning_rate = learning_rate ),
                .metrics   = ((metric_func[]){ get_metric_func( get_loss_name( nn->loss ) ),
                    get_metric_func( "categorical_accuracy" ), NULL }),
                )
            );
#endif
    optimizer_t *opt = OPTIMIZER(
         adam_new(
             nn, 
             OPTIMIZER_PROPERTIES(
                .batchsize = 16,
                .shuffle   = true,
                .metrics   = ((metric_func[]){ get_metric_func( get_loss_name( nn->loss ) ),
                    get_metric_func( "categorical_accuracy" ), NULL }),
                /* .progress  = NULL */
            ),
            ADAM_PROPERTIES( .learning_rate=learning_rate )
         )
    );

    callback_t *callbacks[] = {
        CALLBACK( logger_new( LOGGER_SETTINGS() ) ),
        NULL
    };

    /* Main training loop */
    float results[ 2 * optimizer_get_n_metrics( opt ) ];
    int n_epochs = 2;
    for( int epoch = 0; epoch < n_epochs; epoch++ ){
        optimizer_run_epoch( opt, n_train_samples, train_features, train_labels,
                                  n_test_samples,  test_features,  test_labels, results );
        for( callback_t **cb = callbacks; *cb; cb++ )
            callback_run( *cb, opt, results, true );
    }

    /* Report the final results */
    printf("Train loss    : %5.5f\n", results[0] );
    printf("Train accuracy: %5.5f\n", results[1] );
    printf("Test loss     : %5.5f\n", results[2] );
    printf("Test accuracy : %5.5f\n", results[3] );

    /* Clean up the resources */
    neuralnet_free( nn );
    optimizer_free( opt );

    for( callback_t **cb = callbacks; *cb; cb++ )
        callback_free( *cb );

    free( train_features );
    free( train_labels );
    free( test_features );
    free( test_labels);

    return 0;
}

