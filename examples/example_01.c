#include "npy_array.h"
#include "npy_array_list.h"
#include "neuralnet.h"
#include "simd.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main( int argc, char *argv[] )
{
    /* Read the datafile created in python+numpy */
    npy_array_list_t *filelist = npy_array_list_load( "mushroom_train.npz" );
    assert( filelist );
    
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

    /* Set up a new Neural Network */
    neuralnet_t *nn = neuralnet_create( 3,
            INT_ARRAY( train_X->shape[1], 64, 32, 1 ),
            STR_ARRAY( "relu", "relu", "sigmoid" ) );
    assert( nn );

    neuralnet_initialize( nn, STR_ARRAY("kaiming", "kaiming", "kaiming"));
    neuralnet_set_loss( nn, "binary_crossentropy" );


    /* Training with plain Stochastic Gradient Decsent (SGD) */    
    const int n_train_samples = train_X->shape[0];
    const int n_features      = train_X->shape[1];
    const int n_parameters    = neuralnet_total_n_parameters( nn );
    const float learning_rate = 0.01f;

    float SIMD_ALIGN(gradient[n_parameters]); 

    float *train_feature = (float*) train_X->data;
    float *train_target  = (float*) train_Y->data;
    for( int i = 0; i < n_train_samples; i++ ){
        neuralnet_backpropagation( nn, train_feature, train_target, gradient );
        for( int j = 0; j < n_parameters; j++ )
            gradient[j] *= -learning_rate;
        neuralnet_update( nn, gradient );
        train_feature += n_features;
        train_target  += 1;
    }
    printf("Done SGD training of one epoch. \n # features: %6d \n #  samples: %6d \n", n_features, n_train_samples );

    /* Evaluate the training by calculating the test accuracy */
    int correct_count = 0;
    const int n_test_samples = test_X->shape[0];
    
    float *test_feature = (float*) test_X->data;
    float *test_target  = (float*) test_Y->data;
    for( int i = 0; i < n_test_samples; i++ ){
        float output[1];
        neuralnet_predict( nn, test_feature, output );
        int y_pred = output[0] > 0.5f ? 1 : 0;
        int y_true = *test_target > 0.5f ? 1 : 0;
        if( y_pred == y_true )
            correct_count++;
        test_feature += n_features;
        test_target  += 1;
    }

    printf("Accuracy: %5.5f   ( %d / %d )\n", (float) correct_count / (float) n_test_samples,
         correct_count, n_test_samples );

    /* Clean up the resources */
    neuralnet_free( nn );
    npy_array_list_free( filelist );
    return 0;
}

