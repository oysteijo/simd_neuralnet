#include "npy_array.h"
#include "neuralnet.h"
#include "simd.h"

#include "optimizer.h"
#include "optimizer_implementations.h"
#include "loss.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static uint32_t read_uint_big_endian( FILE *fp )
{
    union { unsigned char c[4]; uint32_t i } val;
    fread( &val, 1, sizeof(val), fp );
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
    fread( magic, 4, sizeof(unsigned char), fp );
    assert( magic[0] == 0 );
    assert( magic[1] == 0 );
    assert( magic[2] == 0x08 );  /* We expect unsigned bytes */

    int n_dim = (int) magic[3];

    /* if this is features we expect n_dim to be three */
    assert( n_dim == 3 );
    printf( "n_dim: %d\n", n_dim );

    /* Reuse the magic to get the right (little endian) integer */
    *n_samples = read_uint_big_endian( fp );
    printf( "read n_samples: %d\n", *n_samples );

    uint32_t n_rows = read_uint_big_endian( fp );
    uint32_t n_cols = read_uint_big_endian( fp );

    *n_features = n_rows * n_cols;

    unsigned char *raw = malloc( *n_samples * *n_features * sizeof( unsigned char ));
    fread( raw, *n_samples * *n_features, sizeof( unsigned char ), fp );

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

float *read_idx_labels( const char *filename, uint32_t *n_samples, uint32_t *n_targets )
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
    fread( magic, 4, sizeof(unsigned char), fp );
    assert( magic[0] == 0 );
    assert( magic[1] == 0 );
    assert( magic[2] == 0x08 );  /* We expect unsigned bytes */

    int n_dim = (int) magic[3];

    /* if this is labels we expect n_dim to be one */
    assert( n_dim == 1 );
    printf( "n_dim: %d\n", n_dim );

    /* Reuse the magic to get the right (little endian) integer */
    *n_samples = read_uint_big_endian( fp );

    printf( "read n_samples: %d\n", *n_samples );

    unsigned char *raw = malloc( *n_samples * sizeof( unsigned char ));
    fread( raw, *n_samples, sizeof( unsigned char ), fp );

    float   *labels = calloc( *n_samples * *n_targets, sizeof( float ));

    /* Do one-hot encoding of target labels */
    for( unsigned int i = 0; i < *n_samples; i++ ){
        *((labels + (i * *n_targets)) + *(raw+i)) = 1.0f;   /* Hehe! Did you get that? */
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
    float *train_labels   = read_idx_labels  ( "train-labels-idx1-ubyte", &n_train_samples, &n_output_targets );
    float *test_features  = read_idx_features( "t10k-images-idx3-ubyte", &n_test_samples, &n_input_features );
    float *test_labels    = read_idx_labels  ( "t10k-labels-idx1-ubyte", &n_test_samples, &n_output_targets );

    if( !train_labels   || !train_features || !test_labels    || !test_features )
        return 0;

    /* Set up a new Neural Network */
    neuralnet_t *nn = neuralnet_create( 5,
            INT_ARRAY( n_input_features, 512, 256, 128, 64, n_output_targets ),
            STR_ARRAY( "relu", "relu", "relu", "relu", "softmax" ) );
    assert( nn );

    neuralnet_initialize( nn, NULL ); 
    neuralnet_set_loss( nn, "categorical_crossentropy" );


    /* Training with plain Stochastic Gradient Decsent (SGD) */    
    const float learning_rate = 0.01f;

    optimizer_t *sgd = optimizer_new( nn, 
            OPTIMIZER_CONFIG(
                .batchsize = 16,
                .shuffle   = true,
                .run_epoch = SGD_run_epoch,
                .settings  = SGD_SETTINGS( .learning_rate = learning_rate ),
                .metrics   = ((metric_func[]){ get_metric_func( get_loss_name( nn->loss ) ),
                    get_metric_func( "categorical_accuracy" ), NULL }),
                )
            );

    float results[ 2 * optimizer_get_n_metrics( sgd ) ];
    optimizer_run_epoch( sgd, n_train_samples, train_features, train_labels,
                              n_test_samples,  test_features, test_labels, results );

    printf("Train loss    : %5.5f\n", results[0] );
    printf("Train accuracy: %5.5f\n", results[1] );
    printf("Test loss     : %5.5f\n", results[2] );
    printf("Test accuracy : %5.5f\n", results[3] );

    /* Clean up the resources */
    neuralnet_free( nn );

    free( train_features );
    free( train_labels );
    free( test_features );
    free( test_labels);

    return 0;
}

