/* optimizer.h - Oystein Schonning-Johansen 2012-2014 
 */

/** \struct _optimizer_t
 * \brief Abstract type/interface type to hold different optimizers

 * Typical usage:
 * This is the typical way of creating an optimizer instance. The code below shows how to instansiate
 * the implementation neuralnet_optimizer_t and composite_optimizer_t.

        optimizer_t *sgd = OPTIMIZER(stochastic_gradient_descent_new( SGD_OPTIMIZER_ARGS("race.weights", "race", "race" )));

 * The interface assures the 'run_epoch' method to be implemented, and this method can be called like:

        optimizer_run_epoch( sgd, nn, gnubg, board, values );

 * There is also a free method for all optimizers and a standard for creating.
 * For convienience, there are two macros defined, OPTIMIZER_DECLARE and
 * OPTIMIZER_DEFINE, for use when generating new implementations. OPTIMIZER_DECLARE
 * is typically used in the .h file of an implementation and OPTIMIZER_DEFINE
 * typically used in a .c file of an implementation.
 */

#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include "neuralnet.h"

#include <stdlib.h>  /* malloc/free in macros */
#include <stdio.h>   /* fprintf in macro */

#define OPTIMIZER(v) (optimizer_t*)(v)

enum {
	OUTPUT_WIN, 
	OUTPUT_WINGAMMON, 
	OUTPUT_WINBACKGAMMON, 
	OUTPUT_LOSEGAMMON, 
	OUTPUT_LOSEBACKGAMMON, 
	N_OUTPUT
};

typedef struct _optimizer_t optimizer_t;
struct _optimizer_t {
	void (*run_epoch) (const optimizer_t *self, const board_t *board, float output[ N_OUTPUT ] );
	void (*free) (optimizer_t *self);
};

static inline void optimizer_evaluate( const optimizer_t *self, const board_t *board, float output[ N_OUTPUT ] ){
	self->run_epoch( self, board, output );
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
static void name ## _evaluate( const optimizer_t *eval, const board_t *b, float *output ); \
DLLEXPORT name ## _t * name ## _new( void * UNUSED(config)) \
{	\
	name ## _t *neweval = malloc( sizeof( name ## _t ) ); \
	if ( !neweval ) {\
		fprintf( stderr ,"Can't allocate memory for '" #name "_t' optimizer type.\n"); \
		return NULL; \
	} \
	neweval->eval.run_epoch = name ## _evaluate; \
	neweval->eval.free = (void(*)(optimizer_t*)) free; \
	__VA_ARGS__ ; \
	return neweval; \
}

#define OPTIMIZER_DECLARE(name) \
typedef struct _ ## name ## _t name ## _t; \
name ## _t * name ## _new( void * config); 

#endif  /* __OPTIMIZER_H__ */
