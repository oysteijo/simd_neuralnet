/* optimizer.h - Oystein Schonning-Johansen 2012-2014 
 */

/** \struct _optimizer_t
 * \brief Abstract type/interface type to hold different optimizers

 * Typical usage:
 * This is the typical way of creating an optimizer instance. The code below shows how to instansiate
 * the implementation neuralnet_optimizer_t and composite_optimizer_t.

        optimizer_t *sgd = OPTIMIZER(stochastic_gradient_descent_new( SGD_OPTIMIZER_ARGS( .learning_rate = 0.01 )));

 * The interface assures the 'run_epoch' method to be implemented, and this method can be called like:

        optimizer_run_epoch( sgd, train_X, train_Y, test_X, test_Y, 1 );

 * There is also a free method for all optimizers and a standard for creating.
 * For convienience, there are two macros defined, OPTIMIZER_DECLARE and
 * OPTIMIZER_DEFINE, for use when generating new implementations. OPTIMIZER_DECLARE
 * is typically used in the .h file of an implementation and OPTIMIZER_DEFINE
 * typically used in a .c file of an implementation.
 */

#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include "neuralnet.h"
#include "loss.h"
#include "metrics.h"

#include <stdlib.h>  /* malloc/free in macros */
#include <stdio.h>   /* fprintf in macro */
#include <stdbool.h>   /* fprintf in macro */

#define OPTIMIZER(v) ((optimizer_t*)(v))

#define MAX_METRICS 5

typedef struct _optimizer_t optimizer_t;
struct _optimizer_t {
    void (*run_epoch)( const optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y,
        const unsigned int n_valid_samples, const float *valid_X, const float *valid_Y, float *result );

	void (*free) (optimizer_t *self);
    
    neuralnet_t *nn;
    unsigned long long int iterations;
    bool     shuffle;
    int      batchsize;
//    void     (*progress) ( char *label, int x, int n ); /* Naa... */
    metric_func metrics[MAX_METRICS + 1];  /* NULL terminated */
};

static inline void optimizer_run_epoch( const optimizer_t *self,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y,
        const unsigned int n_valid_samples, const float *valid_X, const float *valid_Y, float *result )
{
	self->run_epoch( self, n_train_samples, train_X, train_Y, n_valid_samples, valid_X, valid_Y, result );
}

static inline void optimizer_free( optimizer_t *self)
{
	self->free( self );
}


static inline int optimizer_get_n_metrics( const optimizer_t *opt )
{
    int i = 0;
    for ( ; i < MAX_METRICS ; i++ )
        if (!opt->metrics[i])
            break;

    return i;
}

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
static void name ## _run_epoch( const optimizer_t *self, \
        const unsigned int n_train_samples, const float *train_X, const float *train_Y, \
        const unsigned int n_valid_samples, const float *valid_X, const float *valid_Y, float *result );  \
DLLEXPORT name ## _t * name ## _new( neuralnet_t *nn, void * UNUSED(config)) \
{	\
	name ## _t *newopt = malloc( sizeof( name ## _t ) ); \
	if ( !newopt ) {\
		fprintf( stderr ,"Can't allocate memory for '" #name "_t' optimizer type.\n"); \
		return NULL; \
	} \
	newopt->opt.run_epoch = name ## _run_epoch; \
	newopt->opt.free = (void(*)(optimizer_t*)) free; \
    newopt->opt.nn = nn; \
    newopt->opt.iterations = 0; \
    newopt->opt.shuffle = true; \
    newopt->opt.batchsize = 1; \
    newopt->opt.metrics[0] = get_metric_func( get_loss_name( nn->loss ) ); \
    newopt->opt.metrics[1] = NULL; \
	__VA_ARGS__ ; \
	return newopt; \
}

#define OPTIMIZER_DECLARE(name) \
typedef struct _ ## name ## _t name ## _t; \
name ## _t * name ## _new( neuralnet_t *nn, void * config); 

#endif  /* __OPTIMIZER_H__ */
