#include "RMSprop.h"

#include "simd.h" 
#include "matrix_operations.h" 

#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#ifdef __AVX__ 
#include <immintrin.h>
#endif

/*  RMSprop.c  */
struct _RMSprop_t 
{
    optimizer_t opt;
    /* Other data */
    float learning_rate;
    float decay;
    float rho;
    float momentum;
    bool  nesterov;

    /* private stuff - don't touch! */
    unsigned int n_iterations;  
    float *velocity;
    float *r;
};

static void RMSprop_optimizer_init( RMSprop_t *rmsprop, rmsprop_properties_t *properties )
{
    rmsprop_properties_t *props = (rmsprop_properties_t*) properties;

    rmsprop->n_iterations = 0;

    rmsprop->learning_rate = props->learning_rate;
    rmsprop->momentum = props->momentum;
    rmsprop->nesterov = props->nesterov;
    rmsprop->decay = props->decay;
    rmsprop->rho = props->rho;

    const unsigned int n_param = neuralnet_total_n_parameters( OPTIMIZER(rmsprop)->nn );

    rmsprop->velocity   = simd_malloc( n_param * sizeof(float) );
    assert( rmsprop->velocity );
    memset( rmsprop->velocity, 0, n_param * sizeof(float));

    rmsprop->r   = simd_malloc( n_param * sizeof(float) );
    assert( rmsprop->r );
    memset( rmsprop->r, 0, n_param * sizeof(float));
}

static void RMSprop_optimizer_free( optimizer_t *opt )
{
    if( !opt ) return;
    simd_free( RMSPROP_OPTIMIZER(opt)->velocity );
    simd_free( RMSPROP_OPTIMIZER(opt)->r );
}

OPTIMIZER_DEFINE(RMSprop, 
    RMSprop_optimizer_init( newopt, properties );
    newopt->opt.free = RMSprop_optimizer_free;
);

/* FIXME */
/* This static function is shared among RMSprop og Adagrad. Note that I've kept the epsilon outside the sqrt. */
/* Discuss: Should this function be moved somewhere? */
static void compute_update( const int n, float *delta_w, const float *r, const float lr )
{
    const float epsilon = 1.0e-7f; /* Same as default K.epsilon (when backend=TensorFlow) in Keras */
    int i = 0;
    const float *r_ptr = r;
#ifdef __AVX__
    const __m256 lr_v = _mm256_set1_ps(-lr);
    const __m256 eps_v = _mm256_set1_ps(epsilon);
    for( ; i <= ((n)-8) ; i += 8, delta_w += 8, r_ptr += 8 ){
        _mm256_store_ps( delta_w, _mm256_mul_ps( _mm256_load_ps( delta_w ),
                    _mm256_div_ps( lr_v, _mm256_add_ps( eps_v, _mm256_sqrt_ps( _mm256_load_ps( r_ptr ) )))));
    }
#endif
    for( ; i < n; i++)
        *delta_w++ *= -lr / ( epsilon + sqrtf( *r_ptr++ ));
}

void RMSprop_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )
{
    RMSprop_t *rmsprop = RMSPROP_OPTIMIZER( opt ) ;
    // RMSprop_settings_t *RMSprop = (RMSprop_settings_t*) opt->settings;

    neuralnet_t *nn = opt->nn;
    const unsigned int n_parameters = neuralnet_total_n_parameters( nn );

    for ( unsigned int i = 0; i < n_train_samples ;  ){

        /* Apply interim update */
        if ( rmsprop->momentum > 0.0f ){
            vector_scale( n_parameters, rmsprop->velocity, rmsprop->momentum );
            if( rmsprop->nesterov )  
                neuralnet_update( nn, rmsprop->velocity );
        }

        float SIMD_ALIGN(batchgrad[n_parameters]);
        optimizer_calc_batch_gradient( opt, n_train_samples, train_X, train_Y, &i, batchgrad );
        if( opt->progress ) opt->progress( i, n_train_samples, "Train: " );

        
        if (rmsprop->decay > 0.0f )
            rmsprop->learning_rate *= 1.0f / (1.0f + rmsprop->decay * (float) rmsprop->n_iterations);
        rmsprop->n_iterations++;

        float *delta_w = batchgrad;
        float *r = rmsprop->r; 

        float SIMD_ALIGN(g2[n_parameters]);
        vector_square_elements( n_parameters, g2, batchgrad );
        vector_saxpby( n_parameters, r, 1.0f - rmsprop->rho, g2, rmsprop->rho );

        compute_update( n_parameters, delta_w, r, rmsprop->learning_rate /*, epsilon? */ );
        if( rmsprop->momentum > 0.0f )
            vector_accumulate( n_parameters, rmsprop->velocity, delta_w );

        if ( rmsprop->momentum > 0.0f && !rmsprop->nesterov )  /* if nesterov==true we have already updated based on the velocity */
            neuralnet_update( nn, rmsprop->velocity );
        else
            neuralnet_update( nn, delta_w );
    }
}

