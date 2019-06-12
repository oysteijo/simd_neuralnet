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

#if 0
    /* This is debug info... remove in future */
    static bool do_print_settings = true;
    if ( do_print_settings ){
        printf( "learning_rate: %.4f\nmomentum: %.4f\ndecay: %.4f\nNesterov: %s\n",
            sgd->learning_rate, sgd->momentum, sgd->decay, sgd->nesterov ? "True" : "False");
        do_print_settings = false;
    }
#endif
    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        /* Apply interim update */
        if ( sgd->momentum > 0.0f ){
            vector_scale( n_parameters, opt->velocity, sgd->momentum );
            if( sgd->nesterov )  
                neuralnet_update( nn, opt->velocity );
        }

        /* Calculate batch gradient */
        float SIMD_ALIGN(batchgrad[n_parameters]);
        optimizer_calc_batch_gradient( opt, n_train_samples, train_X, train_Y, &i, batchgrad );

        /* Progress callback */
        opt->progress( i, n_train_samples, "Train: " );
        
        /* Learning rate update */
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
