#include "optimizer.h"
#include "simd.h"
#include "evaluate.h"

#include <string.h>
#include <time.h>
#include <assert.h>

static void prepare_shuffle_pivot( optimizer_t *opt, const unsigned n_train_samples )
{
    static unsigned int _n_sample = 0;
    if ( n_train_samples != _n_sample ){
        _n_sample = n_train_samples;
        opt->pivot = realloc( opt->pivot, n_train_samples * sizeof(unsigned int));
        if ( !opt->pivot ){
            fprintf( stderr, "Cannot allocate pivot array.\n");
            return;
        }
        for ( unsigned int i = 0; i < n_train_samples; i++ )
            opt->pivot[i] = i;

        srand( time (NULL ) );
    }
}

static void fisher_yates_shuffle( unsigned int *arr, unsigned int n )
{
    for ( unsigned int i = n-1; i > 0; i-- ){
        unsigned int j = rand() % (i+1);
        unsigned int tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

#define METRIC_LIST(...) ((metric_func[]){ __VA_ARGS__, NULL }) 

optimizer_t *optimizer_new( neuralnet_t *nn, void *data )
{
    optimizer_config_t *conf = (optimizer_config_t*) data;

    optimizer_t *newopt = malloc( sizeof( optimizer_t ));
	if ( !newopt ) {
		fprintf( stderr ,"Can't allocate memory for optimizer_t type.\n");
		return NULL;
	}

    /* First the configs */
    newopt->nn         = nn;
    newopt->batchsize  = conf->batchsize; /* FIXME: Chack sanity */
    newopt->shuffle    = conf->shuffle;
    newopt->metrics    = conf->metrics;   /* Lots of checks can be done */
    newopt->settings   = conf->settings;
    newopt->run_epoch  = conf->run_epoch;
    newopt->progress   = conf->progress;

    newopt->iterations = 0;

    /* now we do the internal data stuff */
    const unsigned int n_param = neuralnet_total_n_parameters( nn );
    newopt->pivot      = NULL; /* This will be allocated in the main loop */


    /* FIXME: Loop? */
    newopt->velocity   = simd_malloc( n_param * sizeof(float) );
    assert( newopt->velocity );
    memset( newopt->velocity, 0, n_param * sizeof(float));

    /* Adam moments */
    newopt->s   = simd_malloc( n_param * sizeof(float) );
    newopt->r   = simd_malloc( n_param * sizeof(float) );
    assert( newopt->s );
    assert( newopt->r );
    memset( newopt->s, 0, n_param * sizeof(float));
    memset( newopt->r, 0, n_param * sizeof(float));


    return newopt;
}

void optimizer_free( optimizer_t *opt )
{
    assert( opt );
    if( opt->pivot )
        free( opt->pivot );
    free( opt->velocity );
    free( opt->s );
    free( opt->r );
    free( opt );
    opt = NULL;
}

void optimizer_calc_batch_gradient( optimizer_t *opt, 
        const unsigned int n_train_samples, const float *train_X, const float *train_Y,
        unsigned int *i, float *batchgrad)
{
    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    /* There is a bug in OpenMP -- if using reduction on an aligned array, there will be threaded
       copies of the arrays in each thread. These copies will not necesarrily be aligned, but rather
       follow the previous in memory. Solving this can be done by padding. I've not seen this documented
       but it seems to work. Hence this padding. */
#ifdef __AVX__
#define PADDING_SIZE 16
    /* Hmmm.... maybe it doesn't work after all? it doesn't work with PADDING_SIZE=32, but
       it works with PADDING_SIZE=16... why? Is this really a stable solution? */
    const unsigned int n_param_padding = n_parameters + (PADDING_SIZE - n_parameters % PADDING_SIZE);
    assert( (n_param_padding % PADDING_SIZE) == 0);
#else
    const unsigned int n_param_padding __attribute__((unused))  = n_parameters;
#endif
    memset( batchgrad, 0, n_parameters * sizeof(float));  /* Clear the batch grad */

    const int n_input  = nn->layer[0].n_input;
    const int n_output = nn->layer[nn->n_layers-1].n_output;

    const int remaining_samples = (int) n_train_samples - (int) *i;
    const int batchsize = remaining_samples < opt->batchsize ? remaining_samples : opt->batchsize;
#pragma omp parallel for reduction(+:batchgrad[0:n_param_padding])
    for ( int b = 0 ; b < batchsize; b++){
        const int idx = *i + b;
        float SIMD_ALIGN(grad[n_parameters]);
        neuralnet_backpropagation( nn, train_X + (opt->pivot[idx] * n_input), train_Y + (opt->pivot[idx] * n_output), grad );
        /* When using OpenMP, OpenMP will not align stack allocated arrays -- we therefore
           have to use `_unaligned` for this accumulation. :-(  */
        vector_accumulate_unaligned( n_parameters, batchgrad, grad );
    }
    *i += batchsize;
    vector_divide_by_scalar( n_parameters, batchgrad, (float) batchsize );
}


void optimizer_run_epoch( optimizer_t *self,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y,
        const unsigned int n_valid_samples, const float *valid_X, const float *valid_Y, float *results )
{
    /* Setup some stuff */
    prepare_shuffle_pivot( self, n_train_samples );
    if( self->shuffle )
        fisher_yates_shuffle( self->pivot, n_train_samples );

    /* Run the epoch */
    assert ( self->run_epoch );
	self->run_epoch(self, n_train_samples, train_X, train_Y );

    /* Calculate the losses */
    /* First the train loss */
    int n_metrics = optimizer_get_n_metrics( self );
    evaluate( self->nn, n_train_samples, train_X, train_Y, self->metrics, results );  

    /* and if validation is given - do it */
    bool has_valid = valid_X && valid_Y && n_valid_samples > 0;
    if( has_valid ){
        evaluate( self->nn, n_valid_samples, valid_X, valid_Y, self->metrics, results + n_metrics );
    }

#if 0
    /* The callback system is still under construction. Please call the callback functions in your main loop */
    /* I think there will be a design change. The callbacks *should* be called in the main loop. That is actually
       better design than having the callbacks a part of the optimizer. After all they are quite independent.
       (Well, at least sort of independent. I guess the callbacks have to know about the optimizer, but the
       optimizer doesn't have to know about the callbacks.) Single responsibility. The optimizer do only
       the optimization, the callbacks do their thing. They do not need to be connected. */
#endif
}

