#include "c_npy.h"
#include "neuralnet.h"
#include "activation.h"
#include "loss.h"
#include "metrics.h"

#include "strtools.h"
#include "progress.h"

#include <cblas.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include <immintrin.h>
#include "simd.h"

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
 * Implements a <- a + scale * b   (Which is saxpy, isn't it?)
 * */
/* Hehe - a call to cblas_saxpy instead of this primitive loop, is actually slower!! */
#if defined(__AVX__)
void scale_and_add_vector( unsigned int n, float *y, const float scale, const float *b )
{
	unsigned int count = n >> 3;
    unsigned int remaining = n & 0x7;
	const float *b_ptr = b;
	float *y_ptr = y; 

    /* y is always aligned, that is not a problem. However, b, which comes from the gradient is not aligned
       when a n_output from a layer is not mod 8. There is of course a possibility to do things sequensial 
       until we are aligned, and then to vectorized operations, however.... fix this later, it won't gain much. */
    if ( simd_aligned( y ) && simd_aligned( b ) ){
        __m256 scalevec = _mm256_set1_ps(scale);
        for (int j = count; j; j--, y_ptr += 8, b_ptr += 8){
#if defined(__AVX2__)
            _mm256_store_ps(y_ptr, _mm256_fmadd_ps( _mm256_load_ps(b_ptr), scalevec, _mm256_load_ps(y_ptr)));
#else
            _mm256_store_ps(y_ptr, _mm256_add_ps(_mm256_load_ps(y_ptr), _mm256_mul_ps(_mm256_load_ps(b_ptr), scalevec)));
#endif
        }
    
    } else {
        count = 0;
        remaining = n;
    }

    /* Do the rest. If the user has done his/her homework, this should not be necesarry */
    if( !remaining ) return;

    y_ptr = y + (count*8);
    b_ptr = b + (count*8);

    for ( unsigned int i = 0; i < remaining; i++ )
        *y_ptr++ += scale * *b_ptr++;
}
#else  /* not __AVX__ */
void scale_and_add_vector( unsigned int n, float *y, const float scale, const float *b )
{
    float *y_ptr = y;
    const float *b_ptr = b;

    for ( unsigned int i = 0; i < n; i++ )
        *y_ptr++ += scale * *b_ptr++;
}
#endif

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

float evaluate( neuralnet_t *nn, cmatrix_t *test_X, cmatrix_t *test_Y, metric_func metric )
{
    float *test_X_ptr = (float*) test_X->data;
    float *test_Y_ptr = (float*) test_Y->data;

    const size_t n_test_samples = test_X->shape[0];
    assert ( n_test_samples == test_Y->shape[0] );

    const int n_input = test_X->shape[1];
    const int n_output = test_Y->shape[1];

    /* OpenMP thread */
    float total_error = 0.0f;
    for ( unsigned int i = 0; i < n_test_samples; i++ ){
        float y_pred[n_output];
        neuralnet_predict( nn, test_X_ptr + (i*n_input), y_pred );
        total_error += metric( n_output, y_pred, test_Y_ptr + (i*n_output));
    }
    return total_error /= (float) n_test_samples;
}

int main( int argc, char *argv[] )
{
    if (argc != 4 ){
        fprintf( stderr, "Usage: %s <weightsfile.npz> <list of activations> <trainsamples.npz>\n", argv[0] );
        return 0;
    }

    cmatrix_t **train_test = c_npy_matrix_array_read( argv[3]);
    if( !train_test ) return -1;

    cmatrix_t *train_X = train_test[0];
    cmatrix_t *train_Y = train_test[1];
    cmatrix_t *test_X = train_test[2];
    cmatrix_t *test_Y = train_test[3];

    float learning_rate = 0.01;


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
    neuralnet_set_loss( nn, "mean_squared_error" );

    // metric_func metric = backgammon_equity_absolute_error;
    metric_func metric = get_metric_func( get_loss_name( nn->loss ));

    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    /* One epoch */
    assert( train_X->shape[0] == train_Y->shape[0] );
    const size_t n_samples = train_X->shape[0];
    
    if( n_samples > RAND_MAX )
        printf("Oh! RAND_MAX is less than the number of samples! Does shuffle work?");

    const int n_input  = train_X->shape[1];
    const int n_output = train_Y->shape[1];

    assert( n_input == nn->layer[0].n_input );
    assert( n_output == nn->layer[nn->n_layers-1].n_output );

    unsigned int *pivot = malloc( n_samples * sizeof(unsigned int) );
    if ( !pivot ){
        fprintf( stderr, "Cannot allocate pivot array.\n");
        return 0;
    }
    for ( unsigned int i = 0; i < n_samples; i++ )
        pivot[i] = i;

    int n_epochs = 20;
    srand( 42 );
    for ( int epoch = 0; epoch < n_epochs; epoch++ ){

        float *train_X_ptr = (float*) train_X->data;
        float *train_Y_ptr = (float*) train_Y->data;

        for ( unsigned int i = 0; i < n_samples; i++ ){

            float SIMD_ALIGN(grad[n_parameters]); /* simd? */
            neuralnet_backpropagation( nn, train_X_ptr + (pivot[i] * n_input), train_Y_ptr + (pivot[i] * n_output), grad );

            /* update */
            const float *ptr = grad;
            for ( int l = 0; l < nn->n_layers; l++ ){
                const int n_inp = nn->layer[l].n_input;
                const int n_out = nn->layer[l].n_output;
                scale_and_add_vector( n_out, nn->layer[l].bias, -learning_rate, ptr );
                // cblas_saxpy( n_out, -learning_rate, ptr, 1, nn->layer[l].bias, 1 );
                ptr += n_out;
                scale_and_add_vector( n_out * n_inp, nn->layer[l].weight, -learning_rate, ptr );
                // cblas_saxpy( n_out * n_inp, -learning_rate, ptr, 1, nn->layer[l].weight, 1 );
                ptr += n_inp * n_out;
            }
            char label[20];
            sprintf(label, "Epoch %2d: ", epoch);
            progress_bar(label, i, n_samples-1 );
        }
        fisher_yates_shuffle( pivot, n_samples );

        /* Now test */
        float *test_X_ptr = (float*) test_X->data;
        float *test_Y_ptr = (float*) test_Y->data;

        const size_t n_test_samples = test_X->shape[0];
        /* OpenMP thread */
        float total_error = 0.0f;
        for ( unsigned int i = 0; i < n_test_samples; i++ ){
            float y_pred[n_output];
            neuralnet_predict( nn, test_X_ptr + (i*n_input), y_pred );
            total_error += metric( n_output, y_pred, test_Y_ptr + (i*n_output));
        }
        total_error /= n_test_samples;
        time_t now = time(NULL);
        struct tm *ltime = localtime( &now );
        char outstr[200];
        strftime( outstr, sizeof(outstr), "%H:%M:%S", ltime );
        printf("%s Epoch: %d  %s: % .7e\n", outstr, epoch, get_metric_name( metric ), total_error );

    }
    free( pivot );

    /* log and report */
    neuralnet_save( nn, "after-20-epochs.npz" );
    neuralnet_free( nn );
    c_npy_matrix_array_free( train_test );
    return 0;
}    
    
