#include "c_npy.h"
#include "neuralnet.h"
#include "activation.h"
#include "loss.h"

#include "strtools.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

STRSPLIT_INIT
// STRSPLIT_LENGTH_INIT
STRSPLIT_FREE_INIT

int main( int argc, char *argv[] )
{
    if (argc != 6 ){
        fprintf( stderr, "Usage: %s <weightsfile.npz> <list of activations> <trainsamples.npz>\n", argv[0] );
        return 0;
    }

    /* It does not handle any whitespace in front of (or trailing) */
    char **split = strsplit( argv[2], ',' );

    neuralnet_t *nn = neuralnet_new( argv[1], split );
    assert( nn );
    strsplit_free(split);

    neuralnet_set_loss( nn, argv[3] );

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
    float grad[neuralnet_total_n_parameters( nn )];
    neuralnet_backpropagation( nn, (float*) inp->data, (float*) target->data, grad );

    float *ptr = grad;
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
