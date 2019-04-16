#include "neuralnet.h"
#include "c_npy.h"
#include <stdio.h>
int main( int argc, char *argv[] )
{
    if( argc != 2)
        return -1;

    neuralnet_t *nn = neuralnet_new( argv[1] );
    if(!nn){
        return -2;
    }

    cmatrix_t *cm = c_npy_matrix_read_file( "random_input.npy");
    if(!cm)
        return -3;

    float pred[4] = { 0 };
    neuralnet_evaluate( nn, (float*) cm->data, pred );
    for (int i = 0; i < 4; i++ )
        printf("%.6f  ", pred[i]);
    printf("\n");

    neuralnet_free (nn );
    c_npy_matrix_free(cm);
    return 0;
}

    

