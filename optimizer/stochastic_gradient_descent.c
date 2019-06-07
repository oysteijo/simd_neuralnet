#include "stochastic_gradient_descent.h"
#include "simd.h"
#include "progress.h"
#include "vector_operations.h"

#include <stdlib.h>   /* malloc/free in macros */
#include <stdio.h>    /* fprintf in macro */
#include <string.h>   /* memset */
#include <immintrin.h>

void SGD_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    sgd_settings_t *sgd = (sgd_settings_t*) opt->settings;

    /* This is debug info... remove in future */
    static bool do_print_settings = true;
    if ( do_print_settings ){
        printf( "learning_rate: %.4f\nmomentum: %.4f\ndecay: %.4f\nNesterov: %s\n",
            sgd->learning_rate, sgd->momentum, sgd->decay, sgd->nesterov ? "True" : "False");
        do_print_settings = false;
    }

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        if ( sgd->momentum > 0.0f ){
            vector_scale( n_parameters, opt->velocity, sgd->momentum );
            if( sgd->nesterov )  
                neuralnet_update( nn, opt->velocity );
        }
        float SIMD_ALIGN(batchgrad[n_parameters]);
        memset( batchgrad, 0, n_parameters * sizeof(float));  /* Clear the batch grad */

        int remaining_samples = (int) n_train_samples - (int) i;
        int max_loop = remaining_samples < opt->batchsize ? remaining_samples : opt->batchsize;
        #pragma omp parallel for shared(i) reduction(+:batchgrad[:])
        for ( int b = 0 ; b < max_loop; b++){
            float SIMD_ALIGN(grad[n_parameters]);
            neuralnet_backpropagation( nn, train_X + (opt->pivot[i] * n_input), train_Y + (opt->pivot[i] * n_output), grad );
            vector_accumulate( n_parameters, batchgrad, grad );
            #pragma omp atomic update
            i++;
        }
        vector_divide_by_scalar( n_parameters, batchgrad, (float) max_loop );
        opt->progress( i, n_train_samples, "Train: " );
        
        /* OK... */
        if (sgd->decay > 0.0f )
            sgd->learning_rate *= 1.0f / (1.0f + sgd->decay * (float) opt->iterations);
        opt->iterations++;

        float *neg_eta_grad = batchgrad;
        vector_scale( n_parameters, neg_eta_grad, -sgd->learning_rate );

        if ( sgd->momentum > 0.0f ){
            /* Compute velocity update */
            vector_accumulate( n_parameters, opt->velocity, neg_eta_grad );
        }
         
        /* Apply update */
        if ( sgd->momentum > 0.0f && !sgd->nesterov )  /* if nesterov==true we have already updated based on the velocity */
            neuralnet_update( nn, opt->velocity );
        else
            neuralnet_update( nn, neg_eta_grad );
    }
}
