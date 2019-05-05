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
#include "c_npy.h"

#include <stdlib.h>  /* malloc/free in macros */
#include <stdio.h>   /* fprintf in macro */

#define OPTIMIZER(v) (optimizer_t*)(v)

typedef struct _optimizer_t optimizer_t;
struct _optimizer_t {
	float (*run_epoch) (const optimizer_t *self, const cmatrix_t *train_X, const cmatrix_t *train_Y, 
                                                const cmatrix_t *test_X, const cmatrix_t *test_Y, unsigned int batch_size );
	void (*free) (optimizer_t *self);
    
    neuralnet_t *nn;
    unsigned long long int iterations;
    bool shuffle;
    void (*progress)( char *label, int x, int n ); 
};

static inline float optimizer_run_epoch( const optimizer_t *self, const cmatrix_t *train_X, const cmatrix_t *train_Y, 
                                                const cmatrix_t *test_X, const cmatrix_t *test_Y, unsigned int batch_size )
{
	return self->run_epoch( self, train_X, train_Y, test_X, test_Y, batch_size );
}

static inline void optimizer_free( optimizer_t *self){
	self->free( self );
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
static float name ## _run_epoch( const optimizer_t *opt, const cmatrix_t *train_X, const cmatrix_t *train_Y, \
                      const cmatrix_t *test_X, const cmatrix_t *test_Y, unsigned int batch_size ); \
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
	__VA_ARGS__ ; \
	return newopt; \
}

#define OPTIMIZER_DECLARE(name) \
typedef struct _ ## name ## _t name ## _t; \
name ## _t * name ## _new( neuralnet_t *nn, void * config); 

#endif  /* __OPTIMIZER_H__ */
