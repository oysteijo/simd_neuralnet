#include "optimizer_implementations.h"
#include "npy_array.h"
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
    float momentum      = 0.0f;
    bool nesterov       = false;

    while (true) {
        static struct option long_options[] =
        {
            {"learning_rate", required_argument, 0, 'l'},
            {"momentum",      required_argument, 0, 'm'},
            {"nesterov",      required_argument, 0, 'n'},
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

            case 'm':
                momentum = strtof( optarg, NULL );
                break;

            case 'n':
                nesterov = string_to_bool( optarg );
        }
    }

    if ( momentum <= 0.0f && nesterov ){
        printf( "Nesterov w/o momentum does not make sense. setting nesterov to false.\n");
        nesterov = false;
    }

    printf( "Learning rate: %5.4f\n", learning_rate );
    printf( "Momentum: %5.4f\n", momentum );
    printf( "Nesterov: %s\n", nesterov ? "True" : "False" );

    npy_array_list_t *filelist = npy_array_list_load( "mushroom_train.npz" );
    
    npy_array_list_t *iter = filelist;
    npy_array_t *train_X = iter->array;  iter = iter->next;
    npy_array_t *train_Y = iter->array;  iter = iter->next;
    npy_array_t *test_X = iter->array;   iter = iter->next;
    npy_array_t *test_Y = iter->array;

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

    neuralnet_t *nn = neuralnet_create( 3,
            INT_ARRAY( train_X->shape[1], 64, 32, 1 ),
            STR_ARRAY( "relu", "relu,", "sigmoid" ) );

    assert( nn );
    neuralnet_initialize( nn, "kaiming", "kaiming", "kaiming" );
    neuralnet_set_loss( nn, "binary_crossentropy" );

    optimizer_t *sgd = optimizer_new( nn, 
            OPTIMIZER_CONFIG(
                .batchsize = 16,
                .shuffle   = false,
                .run_epoch = SGD_run_epoch,
                .settings  = SGD_SETTINGS( .learning_rate = learning_rate, .momentum = momentum, .nesterov = nesterov ),
                .metrics   = ((metric_func[]){ get_metric_func( get_loss_name( nn->loss ) ),
                    get_metric_func( "mean_squared_error"),
                    get_metric_func( "binary_accuracy" ), NULL }),
                .progress  = NULL
                )
            );

    int n_metrics = optimizer_get_n_metrics( sgd );

    int n_epochs = 10;
    
    
    /* Table heading */
    printf("Epoch");
    int longest_metric_name_len = 0;
    for ( int j = 0; j < n_metrics ; j++ ){
        int l = strlen( get_metric_name( sgd->metrics[j] ) );
        if ( l > longest_metric_name_len )
            longest_metric_name_len = l;
    }

    longest_metric_name_len++;
    for ( int j = 0; j < n_metrics ; j++ )
        printf("%*s", longest_metric_name_len, get_metric_name( sgd->metrics[j] ));
    
    for ( int j = 0; j < n_metrics ; j++ )
        printf("%*s", longest_metric_name_len, get_metric_name( sgd->metrics[j] ));

    printf("\n");
    
    for ( int i = 0; i < n_epochs; i++ ){
        float results[2*n_metrics];
        optimizer_run_epoch( sgd, n_train_samples, (float*) train_X->data, (float*) train_Y->data,
                                  n_test_samples, (float*) test_X->data, (float*) test_Y->data, results );
        printf( "%4d ", i);  /* same length as "epoch" */
        for ( int j = 0; j < 2*n_metrics ; j++ )
            printf("%*.7e", longest_metric_name_len, results[j] );
        printf("\n");
    }

    /* log and report */
    neuralnet_save( nn, "after-%d-epochs.npz", n_epochs );
    neuralnet_free( nn );
    free( sgd );
    npy_array_list_free( filelist );
    return 0;
}    
