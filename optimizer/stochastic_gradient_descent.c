#include "stochastic_gradient_descent.h"
#include "progress.h"
#include <stdlib.h>   /* malloc/free in macros */
#include <stdio.h>    /* fprintf in macro */
#include <string.h>   /* memset */

/* HERE IS THE SPECIAL SGD CODE */
void SGD_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    sgd_settings_t *sgd = (sgd_settings_t*) opt->settings;

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        /* Batch start */
        if ( opt->batchsize > 1 )
            memset( opt->batchgrad, 0, n_parameters * sizeof(float));  /* Clear the batch grad */

        int b = 0;
        // printf("batchsize: %d\n", opt->batchsize );
        for ( ; b < opt->batchsize && i < n_train_samples; b++, i++ ){
            neuralnet_backpropagation( nn, train_X + (opt->pivot[i] * n_input), train_Y + (opt->pivot[i] * n_output), opt->grad );
            /* then we add */
            if( opt->batchsize > 1 )
                for ( unsigned int w = 0; w < n_parameters; w++ ) /*  Improve this */
                    opt->batchgrad[w] += opt->grad[w];
        }

        if( opt->batchsize > 1 )
            for ( unsigned int w = 0; w < n_parameters; w++ ) /*  Improve this */
                opt->batchgrad[w] /=  (float) b;
        
        /* OK... */
        if (sgd->decay > 0.0f )
            sgd->learning_rate *= 1.0f / (1.0f + sgd->decay * (float) opt->iterations);
        opt->iterations++;

        neuralnet_update( nn, -sgd->learning_rate, opt->batchsize > 1 ? opt->batchgrad : opt->grad );
#if 1
        /* Argh! The progress bar only prints out at some values if i (there is a mod operation). This
         * call will therefor do nothing for some values of batchsize. */
        progress_bar("Training: ", i, n_train_samples-1 );
#endif
    }
}
