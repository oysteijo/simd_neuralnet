#include "c_npy.h"
#include "neuralnet.h"
#include "activation.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/** 
 * Implements a <- a + scale * b
 * */
void scale_and_add_vector( unsigned int n, float *a, const float scale, const float *b )
{
    float *a_ptr = a;
    const float *b_ptr = b;
    for ( unsigned int i = 0; i < n; i++ )
        a[i] += scale * b[i];
        //*a_ptr++ += scale * *b_ptr++;
}

float metric( unsigned int n, const float *y_pred, const float *y_real )
{
    float err = 2.0f * fabsf( y_pred[0] - y_real[0] );
    for ( unsigned int i = 1; i < n; i++ )
        err = fabsf( y_pred[i] - y_real[i] );
    return err;
}

int main( int argc, char *argv[] )
{
    cmatrix_t **train_test = c_npy_matrix_array_read( "contact_training_and_test.npz" );
    cmatrix_t *train_X = train_test[0];
    cmatrix_t *train_Y = train_test[1];
    cmatrix_t *test_X = train_test[2];
    cmatrix_t *test_Y = train_test[3];

    float learning_rate = 0.5;
    neuralnet_t *nn = neuralnet_new( argv[1] );
    assert( nn );

    for ( int i = 0; i < nn->n_layers; i++ )
        nn->layer[i].activation_func = get_activation_func( "relu" );

    neuralnet_set_loss( nn, "mean_absolute_error" );


    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    /* One epoch */
    assert( train_X->shape[0] == train_Y->shape[0] );
    const size_t n_samples = train_X->shape[0];
    

    const int n_input  = train_X->shape[1];
    const int n_output = train_Y->shape[1];

    assert( n_input == nn->layer[0].n_input );
    assert( n_output == nn->layer[nn->n_layers-1].n_output );

    float *train_X_ptr = (float*) train_X->data;
    float *train_Y_ptr = (float*) train_Y->data;

    for ( unsigned int i = 0; i < n_samples; i++, train_X_ptr += n_input, train_Y_ptr += n_output ){
        float grad[n_parameters]; /* simd? */
        memset( grad, 0, n_parameters * sizeof(float));
        neuralnet_backpropagation( nn, train_X_ptr, train_Y_ptr, grad );

        /* update */
        float *ptr = grad;
        for ( int l = 0; l < nn->n_layers; l++ ){
            const int n_inp = nn->layer[l].n_input;
            const int n_out = nn->layer[l].n_output;
            scale_and_add_vector( n_out, nn->layer[l].bias, -learning_rate, ptr );
            ptr += n_out;
            scale_and_add_vector( n_out * n_inp, nn->layer[l].weight, -learning_rate, ptr );
            ptr += n_inp * n_out;
        }
    }

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
    printf("Metric: % .7e\n", total_error );

    /* log and report */
    neuralnet_save( nn, "best.npz" );
    neuralnet_free( nn );
    c_npy_matrix_array_free( train_test );
    return 0;
}    
    
