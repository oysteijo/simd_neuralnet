/* optimizer.h - Øystein Schønning-Johansen 2019 - 2023 */
/*
  vim: ts=4 sw=4 softtabstop=4 expandtab 
 */

/** \struct _optimizer_t
 * \brief structure to hold all optimizer data. The optimizer data type must be set up with the relevalt
 * parameters through the OPTIMIZER_CONFIG() macro. The structure also keeps a reference to the neural network.
 * 
 * Typical usage:

    optimizer_t * myopt = optimizer_new(
            nn,
            OPTIMIZER_CONFIG(
                .batchsize = 32,
                .shuffle   = true,
                .metrics   = METRIC_LIST(
                    get_metric_func ("mean_absolute_error"),
                    get_metric_func ("mean_squared_error")),
                .callbacks = CALLBACK_LIST( ... ),
                .run_epoch = SGD_run_epoch,
                .setting   = SGD_SETTINGS(
                    .learning_rate = 0.01f,
                    .decay         = 0.0f,
                    .momentum      = 0.9f,
                    .nesterov      = true),
                )
            );


   optimizer_run_epoch( myopt, n_samples, train_X, train_Y );
 */

#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include "neuralnet.h"
#include "metrics.h"
#include "progress.h"

#include <stdlib.h>  /* malloc/free in macros */
#include <stdio.h>   /* fprintf in macro */
#include <stdbool.h>   /* fprintf in macro */

#define OPTIMIZER(v) ((optimizer_t*)(v))

typedef struct _optimizer_t optimizer_t;
typedef void (*epoch_func)( optimizer_t *opt, const unsigned int n_samples, const float *X, const float *Y );
struct _optimizer_t {
    void (*run_epoch)( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y);

    void (*free) (optimizer_t *self);

    neuralnet_t  *nn;
    bool         shuffle;
    int          batchsize;
    void         (*progress)( int x, int n, const char *fmt, ...);
    metric_func  *metrics;  /* NULL terminated */
    unsigned int *pivot;    /* Don't touch! */
};

#if defined(_WIN32) || defined(WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#define OPTIMIZER_DEFINE(name,...) \
static void name ## _run_epoch( optimizer_t *opt, \
        const unsigned int n_train_samples, const float *train_X, const float *train_Y ); \
\
/* Constructor and initialization function  */\
DLLEXPORT name ## _t * name ## _new( neuralnet_t *nn, optimizer_config_t optconf, void *config) \
{   \
    name ## _t *newopt = malloc( sizeof( name ## _t ) ); \
    if ( !newopt ) {\
        fprintf( stderr ,"Can't allocate memory for '" #name "_t' optimizer type.\n"); \
        return NULL; \
    } \
    newopt->opt.run_epoch = name ## _run_epoch; \
    newopt->opt.free = (void(*)(optimizer_t*)) free; \
    \
    /* First the configs */ \
    newopt->opt.nn         = nn; \
    newopt->opt.shuffle    = optconf.shuffle;   \
    newopt->opt.batchsize  = optconf.batchsize; \
    newopt->opt.progress   = optconf.progress;  \
    newopt->opt.metrics    = optconf.metrics;   \
    \
    newopt->opt.pivot      = NULL; /* This will be allocated in the main loop */ \
    \
    __VA_ARGS__ ; \
    return newopt; \
}
#if 0
    /* now we do the internal data stuff */
    const unsigned int n_param = neuralnet_total_n_parameters( nn );


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
}
#endif

#define OPTIMIZER_DECLARE(name) \
typedef struct _ ## name ## _t name ## _t; \
name ## _t * name ## _new( neuralnet_t *nn, optimizer_config_t optconf, void * config); 

void optimizer_calc_batch_gradient( optimizer_t *opt, 
        const unsigned int n_train_samples, const float *train_X, const float *train_Y,
        unsigned int *i, float *batchgrad);

void optimizer_run_epoch( optimizer_t *self,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y,
        const unsigned int n_valid_samples, const float *valid_X, const float *valid_Y, float *result );

typedef struct _optimizer_config_t optimizer_config_t;
struct _optimizer_config_t {
    int batchsize;
    bool shuffle;
    metric_func *metrics;
    void (*progress)( int x, int n, const char *fmt, ...);
} ;
/* These are the default values. The end user should not edit this but "override" at creation */
#define OPTIMIZER_CONFIG(...)  (optimizer_config_t)    \
            { .batchsize = 32,                         \
              .shuffle   = true,                       \
              .metrics   = NULL,                       \
              .progress  = progress_ascii,             \
              __VA_ARGS__ }  

#if 0
optimizer_t *optimizer_new( neuralnet_t *nn, void *data );
void         optimizer_free( optimizer_t *opt );
#endif
static inline int optimizer_get_n_metrics( const optimizer_t *opt )
{
    metric_func *mf_ptr = opt->metrics;
    if(!mf_ptr)
        return 0;

    int n_metrics = 0;    
    while ( *mf_ptr++ )
        n_metrics++;

    return n_metrics;
}
#endif  /* __OPTIMIZER_H__ */
