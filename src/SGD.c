#include "SGD.h"

#include "simd.h" 
#include "matrix_operations.h" 

#include <stdbool.h>
#include <string.h>
#include <assert.h>
/*  SGD.c  */
struct _SGD_t 
{
    optimizer_t opt;
    /* Other data */
    float learning_rate;
    float decay;
    float momentum;
    bool  nesterov;

    /* private stuff - don't touch! */
    unsigned int n_iterations;  
    float *velocity;
};

static void SGD_optimizer_init( SGD_t *sgd, sgd_properties_t *properties )
{
    sgd_properties_t *props = (sgd_properties_t*) properties;

    sgd->n_iterations = 0;

    sgd->learning_rate = props->learning_rate;
    sgd->momentum = props->momentum;
    sgd->nesterov = props->nesterov;
    sgd->decay = props->decay;

    const unsigned int n_param = neuralnet_total_n_parameters( OPTIMIZER(sgd)->nn );

    sgd->velocity   = simd_malloc( n_param * sizeof(float) );
    assert( sgd->velocity );
    memset( sgd->velocity, 0, n_param * sizeof(float));
}

static void SGD_optimizer_free( optimizer_t *opt )
{
    if( !opt ) return;
    free( SGD_OPTIMIZER(opt)->velocity );
    free( opt );
}

OPTIMIZER_DEFINE(SGD, 
    SGD_optimizer_init( newopt, properties );
    newopt->opt.free = SGD_optimizer_free;
);

/* Stochastic Gradient Descent */
void SGD_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    SGD_t *sgd = SGD_OPTIMIZER( opt ) ;
    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        /* Apply interim update */
        if ( sgd->momentum > 0.0f ){
            vector_scale( n_parameters, sgd->velocity, sgd->momentum );
            if( sgd->nesterov )  
                neuralnet_update( nn, sgd->velocity );
        }

        /* Calculate batch gradient */
        float SIMD_ALIGN(batchgrad[n_parameters]);
        optimizer_calc_batch_gradient( opt, n_train_samples, train_X, train_Y, &i, batchgrad );

        /* Progress callback */
        if( opt->progress) opt->progress( i, n_train_samples, "Train: " );
        
        /* Learning rate update */
        if (sgd->decay > 0.0f )
            sgd->learning_rate *= 1.0f / (1.0f + sgd->decay * (float) sgd->n_iterations);
        sgd->n_iterations++;

        float *neg_eta_grad = batchgrad;
        vector_scale( n_parameters, neg_eta_grad, -sgd->learning_rate );

        if ( sgd->momentum > 0.0f ){
            /* Compute velocity update */
            vector_accumulate( n_parameters, sgd->velocity, neg_eta_grad );
        }
         
        /* Apply update */
        if ( sgd->momentum > 0.0f && !sgd->nesterov )  /* if nesterov==true we have already updated based on the velocity */
            neuralnet_update( nn, sgd->velocity );
        else
            neuralnet_update( nn, neg_eta_grad );
    }
}

