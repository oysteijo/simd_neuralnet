#include "neuralnet.h"
#include "activation.h"
#include "c_npy.h"
#include "simd.h"

#include <string.h>
#include <stdio.h>

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
    float *ptr = v;
    printf("[\n");
    for ( int i = 0; i < m; i++ ){
        printf("  ");
        print_vector( n, ptr );
        ptr += n;
    }
    printf("]\n");
}

int main( int argc, char *argv[] )
{
    if( argc != 2)
        return -1;

    neuralnet_t *nn = neuralnet_new( argv[1] );
    if(!nn){
        return -2;
    }

    nn->layer[nn->n_layers-1].activation_func = get_activation_func("softplus");

    cmatrix_t *cm = c_npy_matrix_read_file( "random_input.npy");
    if(!cm)
        return -3;

    float pred[4] = { 0 };
    neuralnet_predict( nn, (float*) cm->data, pred );
    for (int i = 0; i < 4; i++ )
        printf("%.6f  ", pred[i]);
    printf("\n");

#if 1
    int grad_size = 0;
    for ( int i = 0; i < nn->n_layers; i++ )
        grad_size += (nn->layer[i].n_input + 1) * nn->layer[i].n_output;

    float SIMD_ALIGN(grad[grad_size]);
    memset( grad, 0, grad_size * sizeof(float));
    float target[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
    neuralnet_set_loss( nn, "mean_squared_error" );
//    neuralnet_set_loss( nn, "crossentropy" );
    neuralnet_backpropagation( nn, (float*) cm->data, target, grad );
#endif
//    neuralnet_save( nn, "new_net.npz");
    float *ptr = grad;
    for ( int i = 0 ; i < nn->n_layers; i++ ){
        int n_inp = nn->layer[i].n_input;
        int n_out = nn->layer[i].n_output;
        print_vector( n_out, ptr );
        ptr += n_out;
        print_matrix( n_inp, n_out, ptr  );
        ptr += n_inp * n_out;

#if 0
        for ( int j = 0, j < nn->layer[i].n_output; j++)
           *b++ -= lr * *ptr++; 

        for ( int j = 0, j < nn->layer[i].n_output * nn->layer[i].n_input; j++)
           *w++ -= lr * *ptr++;
#endif
    }


    neuralnet_free (nn );



    c_npy_matrix_free(cm);
    return 0;
}

    

