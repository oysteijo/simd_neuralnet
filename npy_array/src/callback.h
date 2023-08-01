/* callback.h - oysteijo@gmail.com  
 */

/** \struct _callback_t
 * \brief Abstract type/interface type to hold different callbacks

 * Typical usage:
 * This is the typical way of creating an callback instance. The code below shows how to instansiate
 * the implementation neuralnet_callback_t and composite_callback_t.

        callback_t *log            = CALLBACK(logging_new( LOGGING_ARGS( ) ));
        callback_t *earlystopping  = CALLBACK(earlystopping_new( EARLYSTOPPING_ARGS( )));

 * The interface assures the 'callback_run' method to be implemented, and this method can be called like:

        callback_run( log, opt, result, has_valid );

 * There is also a free method for all callbacks and a standard for creating.
 * For convienience, there are two macros defined, CALLBACK_DECLARE and
 * CALLBACK_DEFINE, for use when generating new implementations. CALLBACK_DECLARE
 * is typically used in the .h file of an implementation and CALLBACK_DEFINE
 * typically used in a .c file of an implementation.
 */

#ifndef __CALLBACK_H__
#define __CALLBACK_H__

#include "optimizer.h"

#include <stdlib.h>  /* malloc/free in macros */
#include <stdio.h>   /* fprintf in macro */
#include <stdbool.h>   /* fprintf in macro */

#define CALLBACK(v) (callback_t*)(v)

// typedef struct _optimizer_t optimizer_t; /* A 'forward' from optimizer.h */

typedef struct _callback_t callback_t;
struct _callback_t {
	void (*callback_run) ( callback_t *self, optimizer_t *opt, const float *result, bool has_valid );
	void (*free) (callback_t *self);
};

static inline void callback_run( callback_t *self, optimizer_t *opt, const float *result, bool has_valid )
{
	self->callback_run( self, opt, result, has_valid );
}

static inline void callback_free( callback_t *self)
{
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

#define CALLBACK_DEFINE(name,...) \
static void name ## _callback_run( callback_t *cb, optimizer_t *opt, const float *result, bool has_valid ); \
DLLEXPORT name ## _t * name ## _new( void * UNUSED(config)) \
{	\
	name ## _t *newcb = malloc( sizeof( name ## _t ) ); \
	if ( !newcb ) {\
		fprintf( stderr ,"Can't allocate memory for '" #name "_t' callback type.\n"); \
		return NULL; \
	} \
	newcb->cb.callback_run = name ## _callback_run; \
	newcb->cb.free = (void(*)(callback_t*)) free; \
	__VA_ARGS__ ; \
	return newcb; \
}

#define CALLBACK_DECLARE(name) \
typedef struct _ ## name ## _t name ## _t; \
name ## _t * name ## _new( void * config); 

#endif  /* __CALLBACK_H__ */
