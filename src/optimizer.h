/* optimizer.h - Øystein Schønning-Johansen 2019 - 2023 */
/*
  vim: ts=4 sw=4 softtabstop=4 expandtab 
 */

/* FIXME: write some documentation */

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
	int          n_metrics;
    unsigned int *pivot;    /* Don't touch! */
};

#if defined(__GNUC__)
#define UNUSED(c) c __attribute__((__unused__))
#else
#define UNUSED(c)
#endif

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
DLLEXPORT name ## _t * name ## _new( neuralnet_t *nn, optimizer_properties_t optconf, void UNUSED(*properties)) \
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
    newopt->opt.n_metrics  = 0;                 \
    \
    newopt->opt.pivot      = NULL; /* This will be allocated in the main loop */ \
    \
    metric_func *mf_ptr = optconf.metrics; \
    if(!mf_ptr) \
        newopt->opt.n_metrics = 0; \
	else while ( *mf_ptr++ ) \
        newopt->opt.n_metrics++; \
	newopt->opt.metrics = malloc( (newopt->opt.n_metrics+1) * sizeof(metric_func)); \
	memcpy( newopt->opt.metrics, optconf.metrics, (newopt->opt.n_metrics+1) * sizeof( metric_func )); \
    __VA_ARGS__ ; \
    optimizer_check_sanity( OPTIMIZER(newopt) ); \
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
name ## _t * name ## _new( neuralnet_t *nn, optimizer_properties_t optconf, void * config); 

typedef struct _optimizer_properties_t optimizer_properties_t;
struct _optimizer_properties_t {
    int batchsize;
    bool shuffle;
    metric_func *metrics;
    void (*progress)( int x, int n, const char *fmt, ...);
};

/* These are the default values. The end user should not edit this but "override" at creation */
#define OPTIMIZER_PROPERTIES(...)  (optimizer_properties_t)    \
            { .batchsize = 32,                         \
              .shuffle   = true,                       \
              .metrics   = NULL,                       \
              .progress  = progress_ascii,             \
              __VA_ARGS__ }  

void optimizer_calc_batch_gradient( optimizer_t *opt, 
        const unsigned int n_train_samples, const float *train_X, const float *train_Y,
        unsigned int *i, float *batchgrad);

void optimizer_run_epoch( optimizer_t *self,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y,
        const unsigned int n_valid_samples, const float *valid_X, const float *valid_Y, float *result );

void optimizer_check_sanity( optimizer_t * opt);

static inline void optimizer_free( optimizer_t *opt )
{
    opt->free( opt );
    if ( opt->metrics )
        free( opt->metrics );
    if ( opt->pivot )
        free( opt->pivot );
    free( opt );
}

static inline int optimizer_get_n_metrics( const optimizer_t *opt )
{
    return opt->n_metrics;
}
#endif  /* __OPTIMIZER_H__ */
