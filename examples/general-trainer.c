#include "npy_array_list.h"
#include "neuralnet.h"
#include "metrics.h"
#include "loss.h"
#include "optimizer.h"
#include "optimizer_implementations.h"

#include "callback.h"
#include "earlystopping.h"
#include "logger.h"
#include "modelcheckpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <assert.h>

enum {
    CB_MODEL_CHECKPOINT,
    CB_LOGGER,
    CB_EARLY_STOPPING,
    N_CALLBACKS 
};

int main(int argc, char *argv[]) {
    // Declare variables
    int opt;
    char *trainset = NULL;
    char *verificationset = NULL;
    char *neuralnet = NULL;
    char *log = NULL;
    char *optimizer = NULL;
    float learning_rate = 0.001f;
    char *model_checkpoint = NULL;
    char *loss = NULL;
    char *metrics = NULL;
    int batch_size;
    int n_epochs = 10;
    int patience, early_stopping_idx, metric_idx, greater_is_better;

    // Set default values
    batch_size = 32;
    optimizer = "adamw";
    greater_is_better = 1;

    // Define long options
    struct option long_options[] = {
        {"trainset", required_argument, 0, 't'},
        {"verificationset", required_argument, 0, 'v'},
        {"neuralnet", required_argument, 0, 'n'},
        {"loss", required_argument, 0, 'l'},
        {"metrics", required_argument, 0, 'm'},
        {"optimizer", required_argument, 0, 'o'},
        {"learning-rate", required_argument, 0, 'a'},
        {"batch-size", required_argument, 0, 'b'},
        {"log", required_argument, 0, 'q'},
        {"early-stopping", required_argument, 0, 'e'},
        {"model-checkpoint", required_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    const char *short_options = "t:v:n:l:o:a:b:q:e:s:";

    // Parse command-line options
    while ((opt = getopt_long(argc, argv, short_options, long_options, NULL)) != -1) {
        switch (opt) {
            case 't': // Trainset
                trainset = optarg;
                break;
            case 'v': // Verificationset
                verificationset = optarg;
                break;
            case 'n': // Neuralnet
                neuralnet = optarg;
                break;
            case 'l': // Loss
                loss = optarg;
                break;
            case 'm': // Loss
                metrics = optarg;
                break;
            case 'o': // Optimizer
                optimizer = optarg;
                break;
            case 'a': // Optimizer
                learning_rate = strtof(optarg, NULL); /* FIXME: Check error */
                break;
            case 'b': // Batch Size
                batch_size = atoi(optarg);
                break;
            case 'q': // Log
                log = optarg;
                break;
            case 'e': // Early Stopping
                sscanf(optarg, "%d,%d,%d", &patience, &early_stopping_idx, &greater_is_better);
                break;
            case 's': // Model Check Point
                sscanf(optarg, "%s,%d,%d", model_checkpoint, &metric_idx, &greater_is_better);
                break;
            case 'g': // Debugging
                printf("Trainset: %s\n", trainset);
                printf("Verificationset: %s\n", verificationset);
                printf("Neuralnet: %s\n", neuralnet);
                printf("Loss: %s\n", loss);
                printf("Optimizer: %s\n", optimizer);
                printf("Batch Size: %d\n", batch_size);
                printf("Log: %s\n", log);
                printf("Patience: %d\n", patience);
                printf("Early Stopping Index: %d\n", early_stopping_idx);
                printf("Greater is better: %d\n", greater_is_better);
                printf("Model checkpoint: %s\n", model_checkpoint);
                printf("Metric Index: %d\n", metric_idx);
                exit(EXIT_SUCCESS);
            case '?':
                fprintf(stderr, "Usage: %s -t trainset.npz -v verificationset.npz -n neuralnet.npz -l loss -o optimizer -b batch_size -q log_filename -e patience,idx,greater_is_better -s filename,idx,greater_is_better\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    // Print program output
    printf("Trainset: %s\n", trainset);
    printf("Verificationset: %s\n", verificationset);
    printf("Neuralnet: %s\n", neuralnet);
    printf("Loss: %s\n", loss);
    printf("Optimizer: %s\n", optimizer);
    printf("Batch Size: %d\n", batch_size);
    printf("Log: %s\n", log);
    printf("Patience: %d\n", patience);
    printf("Early Stopping Index: %d\n", early_stopping_idx);
    printf("Greater is better: %d\n", greater_is_better);
    printf("Model checkpoint: %s\n", model_checkpoint);
    printf("Metric Index: %d\n", metric_idx);

    /* Read the train data */
    npy_array_list_t *traindata = npy_array_list_load( trainset );
    assert( traindata );
    npy_array_list_t *iter = traindata;
    npy_array_t *train_X = iter->array;  iter = iter->next;
    npy_array_t *train_Y = iter->array;  iter = iter->next;
    assert( train_X->fortran_order == false );
    assert( train_Y->fortran_order == false );

    /* Assert the number of training samples is the same as the number of targets */
    assert( train_X->shape[0] == train_Y->shape[0]);

    /* Read the verify data */
    npy_array_list_t *verifydata = npy_array_list_load( verificationset );
    assert( verifydata );
    iter = verifydata;
    npy_array_t *verify_X = iter->array;  iter = iter->next;
    npy_array_t *verify_Y = iter->array;  iter = iter->next;
    assert( verify_X->fortran_order == false );
    assert( verify_Y->fortran_order == false );

    /* Assert the number of verifying samples is the same as the number of targets */
    assert( verify_X->shape[0] == verify_Y->shape[0]);

    /* assert that the input/output sizes are the same in train and verification */
    assert( train_X->shape[1] == verify_X->shape[1]);
    assert( train_Y->shape[1] == verify_Y->shape[1]);

    const int n_train_samples = train_X->shape[0];
    const int n_verify_samples = verify_X->shape[0];

    /* Read the neural network from file */
    neuralnet_t *nn = neuralnet_load( neuralnet );
    assert( nn );
    neuralnet_set_loss( nn, loss );

    assert( nn->layer[0].n_input == (int) train_X->shape[1] );
    assert( nn->layer[nn->n_layers-1].n_output == (int) train_Y->shape[1] );
    
    /* Metrics */
    /* Here there will be some logic. */

    /* Optimizer logic */
    optimizer_t *optim = optimizer_new( nn, 
            OPTIMIZER_CONFIG(
                .batchsize = batch_size,
                .shuffle   = true,
                .run_epoch = SGD_run_epoch,
                .settings  = SGD_SETTINGS( .learning_rate = learning_rate ),
//                .run_epoch = adamw_run_epoch,
//                .settings  = ADAMW_SETTINGS( .learning_rate = learning_rate ),
                .metrics   = ((metric_func[]){ get_metric_func( get_loss_name( nn->loss ) ),
                    get_metric_func( metrics ), 
                    NULL }),
                .progress  = NULL
                )
            );

    int n_metrics = optimizer_get_n_metrics( optim );
#if 0
    /* Table heading */
    printf("Epoch");
    int longest_metric_name_len = 0;
    for ( int j = 0; j < n_metrics ; j++ ){
        int l = strlen( get_metric_name( optim->metrics[j] ) );
        if ( l > longest_metric_name_len )
            longest_metric_name_len = l;
    }

    longest_metric_name_len++;
    for ( int j = 0; j < n_metrics ; j++ )
        printf("%*s", longest_metric_name_len, get_metric_name( optim->metrics[j] ));
    
    for ( int j = 0; j < n_metrics ; j++ )
        printf("%*s", longest_metric_name_len, get_metric_name( optim->metrics[j] ));

    printf("\n");
#endif
    /* Callbacks */
    callback_t *callbacks[N_CALLBACKS];
    /* callback  Model checkpoint  */
    callbacks[CB_MODEL_CHECKPOINT] = CALLBACK(modelcheckpoint_new( MODELCHECKPOINT_SETTINGS(  ) ));
    /* callback  Log  */
    callbacks[CB_LOGGER]           = CALLBACK(logger_new( LOGGER_SETTINGS( ) ));
    /* callback  Early stopping  */
    callbacks[CB_EARLY_STOPPING]   = CALLBACK(earlystopping_new( EARLYSTOPPING_SETTINGS( )));
    
    /* The main loop */
    for ( int i = 0; i < n_epochs || n_epochs == -1; i++ ){
        float results[2*n_metrics];
        optimizer_run_epoch( optim, n_train_samples, (float*) train_X->data, (float*) train_Y->data,
                                  n_verify_samples, (float*) verify_X->data, (float*) verify_Y->data, results );

        for ( int cb_idx = 0; cb_idx < N_CALLBACKS; cb_idx++ ){
            callback_t *cb = callbacks[cb_idx];
            if(!cb) continue;
            callback_run( cb, OPTIMIZER(optim), results, true );
        }
        if ( earlystopping_do_stop( EARLYSTOPPING(callbacks[CB_EARLY_STOPPING]) ) )
            break;
    }
    neuralnet_free( nn );
    free( optim );

    npy_array_list_free( traindata );
    npy_array_list_free( verifydata );

    return 0;
}
