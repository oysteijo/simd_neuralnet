#include "adagrad.h"
#include "c_npy.h"
#include "neuralnet.h"
#include "activation.h"
#include "loss.h"
#include "metrics.h"

// #include "strtools.h"
#include "progress.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include <getopt.h>

// STRSPLIT_INIT
// STRSPLIT_LENGTH_INIT
// STRSPLIT_FREE_INIT

static bool string_to_bool( const char *str )
{
    /* FIXME : a set/list of strings to try */
    if( !strncmp( str, "true", 4 ) )
        return true;

    if( !strncmp( str, "True", 4 ) )
        return true;

    if( !strncmp( str, "Yes", 3 ) )
        return true;

    if( !strncmp( str, "1", 1 ) )
        return true;

    return false;
}

int main( int argc, char *argv[] )
{
    float learning_rate = 0.01f;

    while (true) {
        static struct option long_options[] =
        {
            {"learning_rate", required_argument, 0, 'l'},
            {0, 0, 0, 0}
        };
        /* getopt_long stores the option index here. */
        int option_index = 0;
        int c = getopt_long (argc, argv, "",
                long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c)
        {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if (long_options[option_index].flag != 0)
                    break;
                printf ("option %s", long_options[option_index].name);
                if (optarg)
                    printf (" with arg %s", optarg);
                printf ("\n");
                break;

            case 'l':
                learning_rate = strtof( optarg, NULL );
                break;
        }
    }

    printf( "Learning rate: %5.4f\n", learning_rate );

    cmatrix_t **train_test = c_npy_matrix_array_read( "mushroom_train.npz" );
    if( !train_test ) return -1;

    cmatrix_t *train_X = train_test[0];
    cmatrix_t *train_Y = train_test[1];
    cmatrix_t *test_X = train_test[2];
    cmatrix_t *test_Y = train_test[3];

    /* if any of these asserts fails, try to open the weight in python and save with
     * np.ascontiguousarray( matrix ) */
    assert( train_X->fortran_order == false );
    assert( train_Y->fortran_order == false );
    assert( test_X->fortran_order == false );
    assert( test_Y->fortran_order == false );

    const int n_train_samples = train_X->shape[0];
    const int n_test_samples = test_X->shape[0];

    printf( "train_X shape: (%zu, %zu)\n", train_X->shape[0], train_X->shape[1] );
    printf( "train_Y shape: (%zu, %zu)\n", train_Y->shape[0], train_Y->shape[1] );
    printf( "test_X shape: (%zu, %zu)\n", test_X->shape[0], test_X->shape[1] );
    printf( "test_Y shape: (%zu, %zu)\n", test_Y->shape[0], test_Y->shape[1] );
    /* It does not handle any whitespace in front of (or trailing) */
    neuralnet_t *nn = neuralnet_new( "initial_mushroom_111_32_1.npz", (char*[]){ "relu", "sigmoid"} );
    assert( nn );
    neuralnet_set_loss( nn, "binary_crossentropy" );

    optimizer_t *adagrad = optimizer_new( nn, 
            OPTIMIZER_CONFIG(
                .batchsize = 16,
                .shuffle   = false,
                .run_epoch = adagrad_run_epoch,
                .settings  = ADAGRAD_SETTINGS( .learning_rate = learning_rate ),
                .metrics   = ((metric_func[]){ get_metric_func( get_loss_name( nn->loss ) ),
                    get_metric_func( "mean_squared_error"),
                    get_metric_func( "binary_accuracy" ), NULL }),
                .progress  = NULL
                )
            );

    int n_metrics = optimizer_get_n_metrics( adagrad );

    int n_epochs = 10;
    
    float results[2*n_metrics];
    
    for ( int i = 0; i < n_epochs; i++ ){
        optimizer_run_epoch( adagrad, n_train_samples, (float*) train_X->data, (float*) train_Y->data,
                                  n_test_samples, (float*) test_X->data, (float*) test_Y->data, results );
        printf( " %3d", i);
        for ( int j = 0; j < 2*n_metrics ; j++ )
            printf("\t%7e", results[j] );
        printf("\n");
    }

    neuralnet_save( nn, "after-%d-epochs.npz", n_epochs );
    neuralnet_free( nn );
    free( adagrad );
    c_npy_matrix_array_free( train_test );
    return 0;
}    
    
