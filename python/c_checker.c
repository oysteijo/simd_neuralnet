#include "c_npy.h"
#include "neuralnet.h"
#include "activation.h"
#include "loss.h"
#include "metrics.h"

#include "strtools.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <immintrin.h>

#include "../neuralnet/simd.h"

/* This is used for debug */
static void print_vector( int n, const float *v )
{
    printf("[ ");
    for (int i = 0; i < n; i++ )
        printf("% .7e ", v[i] );
    printf("]\n");
}

static void print_matrix( int m, int n, const float *v )
{
    const float *ptr = v;
    printf("[\n");
    for ( int i = 0; i < m; i++ ){
        printf("  ");
        print_vector( n, ptr );
        ptr += n;
    }
    printf("]\n");
}

static void print_gradient( const neuralnet_t *nn, const float *grad )
{
    const float *ptr = grad;
    for ( int i = 0 ; i < nn->n_layers; i++ ){
        int n_inp = nn->layer[i].n_input;
        int n_out = nn->layer[i].n_output;
        print_vector( n_out, ptr );
        ptr += n_out;
        print_matrix( n_inp, n_out, ptr  );
        ptr += n_inp * n_out;
    }
}

/** 
 * Implements a <- a + scale * b
 * */
void scale_and_add_vector( unsigned int n, float *a, const float scale, const float *b )
{
    float *a_ptr = a;
    const float *b_ptr = b;
    for ( unsigned int i = 0; i < n; i++ )
        *a_ptr++ += scale * *b_ptr++;
}

typedef float (*metric_func)      (unsigned int n, const float *y_pred, const float *y_real );

static float backgammon_scaled_absolute_error( unsigned int n, const float *y_pred, const float *y_real )
{
    assert( n == 5 );
    float err = 2.0f * fabsf( y_pred[0] - y_real[0] );
    for ( unsigned int i = 1; i < n; i++ )
        err = fabsf( y_pred[i] - y_real[i] );
    return err;
}

static float equity( const float *y )
{
    float scale[5] = {2.0f, 1.0f, 1.0f, -1.0f, -1.0f};  /* Cubeless money game weights */
    float eq = -1.0f;

    for ( int i = 0; i < 5; i++ )
        eq += y[i] * scale[i];
    return eq;
}

static float backgammon_equity_absolute_error( unsigned int n, const float *y_pred, const float *y_real )
{
    assert( n == 5 );
    return fabsf( equity( y_pred ) - equity(y_real ));
}

static void fisher_yates_shuffle( unsigned int *arr, unsigned int n )
{
    for ( unsigned int i = n-1; i > 0; i-- ){
        unsigned int j = rand() % (i+1);
        unsigned int tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

STRSPLIT_INIT
// STRSPLIT_LENGTH_INIT
STRSPLIT_FREE_INIT

int main( int argc, char *argv[] )
{
    if (argc != 6 ){
        fprintf( stderr, "Usage: %s <weightsfile.npz> <quoted list of activations> <trainsamples.npz>\n", argv[0] );
        return 0;
    }

    /*
    cmatrix_t **train_test = c_npy_matrix_array_read( argv[3]);
    if( !train_test ) return -1;

    cmatrix_t *train_X = train_test[0];
    cmatrix_t *train_Y = train_test[1];
    cmatrix_t *test_X = train_test[2];
    cmatrix_t *test_Y = train_test[3];

    float learning_rate = 0.01;
    */


    /* It does not handle any whitespace in front of (or trailing) */
    char **split = strsplit( argv[2], ',' );
#if 0
    for ( int i = 0; i < 3; i++ ){
        printf("%d:%s\n", i, split[i] );
    }
#endif

    neuralnet_t *nn = neuralnet_new( argv[1], split );
    assert( nn );
    strsplit_free(split);

#if 0
    for ( int i = 0; i < nn->n_layers; i++ )
        nn->layer[i].activation_func = get_activation_func( "relu" );
    nn->layer[nn->n_layers-1].activation_func = get_activation_func( "sigmoid" );
#endif
    neuralnet_set_loss( nn, argv[3] );

    // metric_func metric = backgammon_equity_absolute_error;
    metric_func metric = get_metric_func( get_loss_name( nn->loss ));

    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    cmatrix_t *inp    = c_npy_matrix_read_file( argv[ 4 ] );
    cmatrix_t *target = c_npy_matrix_read_file( argv[ 5 ] );

    float prediction[target->shape[0]];
    cmatrix_t save = {
        .data  = (char *) prediction,
        .shape = { 1, target->shape[0], 0 },
        .ndim  = 2,
        .endianness = target->endianness,
        .typechar = target->typechar,
        .elem_size = target->elem_size,
        .fortran_order = target->fortran_order
    };

    neuralnet_predict( nn, (float *) inp->data, prediction );
    c_npy_matrix_write_file( "c_prediction.npy", &save );

    /* Backprop */
    float SIMD_ALIGN(grad[n_parameters]);
    neuralnet_backpropagation( nn, (float*) inp->data, (float*) target->data, grad );

    float *ptr = grad;
    for ( int i = 0 ; i < n_parameters; i++ )
        if ( isnan( *ptr++ ))
            printf("NaN found!\n");

    ptr = grad;
    for ( int l = 0; l < nn->n_layers; l++ ){
        const int n_inp = nn->layer[l].n_input;
        const int n_out = nn->layer[l].n_output;
        save.data = (char*) ptr;
        save.shape[0] = n_out;
        save.ndim = 1;
        char filename[32];
        sprintf(filename, "bias_grad_%d.npy", l );
        c_npy_matrix_write_file( filename, &save );
        ptr += n_out;

        save.data = (char*) ptr;
        save.shape[0] = n_inp;
        save.shape[1] = n_out;
        save.ndim = 2;
        sprintf(filename, "weight_grad_%d.npy", l );
        c_npy_matrix_write_file( filename, &save );
        ptr += n_inp * n_out;
    }


    /* Cleanup */
    c_npy_matrix_free( inp );
    c_npy_matrix_free( target );

    neuralnet_free( nn );
    return 0;
}
