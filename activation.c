#include "activation.h"

#include <string.h>
#include <math.h>

/*
Just thinking out here.... This code needs a cleanup. It started as a implementation
from 'octopus', but at this stage we rather want things to be general than superfast.

Browsing the documentation from an other neural network library (Keras), I see that
the provided activation functions are:

sigmoid
softmax
elu      (Uses a parameter)
selu     (Uses a parameter or two?)
softplus
softsign (What is the derivative?)
relu
tanh
hard_sigmoid (What is the derivative?)
exponential
linear

Advanced activation functions
-----------------------------
LeakyReLU (Uses a parameter)
PReLU     (Uses a trainable parameter!)

OK! Since C does not support closures to generate functions, I guess I will have to wait with activations
using a parameter. (I have to hide the parameter inside a struct and return a pointer to this struct. The
struct will contain the parameter and the function pointer. Let's wait with this.... )

This leaves us with:
sigmoid, softmax, softplus, softsign, relu, tanh, hard_sigmoid, exponential, linear

As a starting point we do this plain with no tricks like SIMD or lookup tables.
*/

static void softplus    ( unsigned int n, float *ar );
static void softsign    ( unsigned int n, float *ar );
static void hard_sigmoid( unsigned int n, float *ar );
static void exponential ( unsigned int n, float *ar );
static void linear      ( unsigned int n, float *ar );
static void relu        ( unsigned int n, float *ar );
static void softmax     ( unsigned int n, float *ar );
static void sigmoid     ( unsigned int n, float *ar );
/* Argh! "tanh" is already defined in math.h  (C99). We give this a different name */
static void tanh_act    ( unsigned int n, float *ar );

#if defined(TRAINING_FEATURES)
static void softplus_derivative    ( unsigned int n, const float *activation, float *ar );
static void softsign_derivative    ( unsigned int n, const float *activation, float *ar );
static void hard_sigmoid_derivative( unsigned int n, const float *activation, float *ar );
static void exponential_derivative ( unsigned int n, const float *activation, float *ar );
static void linear_derivative      ( unsigned int n, const float *activation, float *ar );
static void relu_derivative        ( unsigned int n, const float *activation, float *ar );
static void softmax_derivative     ( unsigned int n, const float *activation, float *ar );
static void sigmoid_derivative     ( unsigned int n, const float *activation, float *ar );
static void tanh_act_derivative    ( unsigned int n, const float *activation, float *ar );
#endif

#define CHECK_ACTIVATION_NAME(func) \
		!strcmp( name, #func) ? func :

activation_func get_activation_func( const char * name ){
	return
        CHECK_ACTIVATION_NAME(softplus)
        CHECK_ACTIVATION_NAME(softsign)
        CHECK_ACTIVATION_NAME(hard_sigmoid)
        CHECK_ACTIVATION_NAME(exponential)
        CHECK_ACTIVATION_NAME(linear)
        CHECK_ACTIVATION_NAME(relu)
        CHECK_ACTIVATION_NAME(softmax)
        CHECK_ACTIVATION_NAME(sigmoid)
		!strcmp( name, "tanh") ? tanh_act :
		NULL;
}

#define CHECK_ACTIVATION_PTR(func) \
        ptr == func ? #func :

const char * get_activation_name( activation_func ptr ){
	return
        CHECK_ACTIVATION_PTR(softplus)
        CHECK_ACTIVATION_PTR(softsign)
        CHECK_ACTIVATION_PTR(hard_sigmoid)
        CHECK_ACTIVATION_PTR(exponential)
        CHECK_ACTIVATION_PTR(linear)
        CHECK_ACTIVATION_PTR(relu)
        CHECK_ACTIVATION_PTR(softmax)
        CHECK_ACTIVATION_PTR(sigmoid)
        ptr == tanh_act ? "tanh" :
		"(unknown)";
}
#undef CHECK_ACTIVATION_NAME
#undef CHECK_ACTIVATION_PTR

#if defined(TRAINING_FEATURES)
#define CHECK_ACTIVATION_DERIV_PTR(func) \
        ptr == func ? func ## _derivative :

activation_derivative get_activation_derivative( activation_func ptr ){
	return
        CHECK_ACTIVATION_DERIV_PTR(softplus)
        CHECK_ACTIVATION_DERIV_PTR(softsign)
        CHECK_ACTIVATION_DERIV_PTR(hard_sigmoid)
        CHECK_ACTIVATION_DERIV_PTR(exponential)
        CHECK_ACTIVATION_DERIV_PTR(linear)
        CHECK_ACTIVATION_DERIV_PTR(relu)
        CHECK_ACTIVATION_DERIV_PTR(softmax)
        CHECK_ACTIVATION_DERIV_PTR(sigmoid)
        CHECK_ACTIVATION_DERIV_PTR(tanh_act)
		NULL;
}
#undef CHECK_ACTIVATION_DERIV_PTR
#endif

static void softmax( unsigned int n, float *ar )
{
    /* FIXME: This might overflow! */
	float sum = 0.0f;
	for ( unsigned int j = 0 ; j < n; j++ ){
		ar[j] = expf( ar[j] );
		sum += ar[j];
	}
	for ( unsigned int j = 0 ; j < n; j++ ){
		ar[j] /= sum;
	}
}

static void relu( unsigned int n, float *ar )
{
	for( unsigned int i = 0; i < n; i++ )
		ar[i] = fmaxf(0.0f, ar[i]);
}

static void sigmoid( unsigned int n, float *ar )
{
    /* Use 0.5 + 0.5*tanhf(0.5*x) ?? */
	for( unsigned int i = 0; i < n; i++ )
		ar[i] = 1.0f / (1.0f + expf(-ar[i]));
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
static void linear( unsigned int n, float *ar )
{
    /* Do nothing */
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static void tanh_act( unsigned int n, float *ar )
{
	for( unsigned int i = 0; i < n; i++ )
		ar[i] = tanhf(ar[i]);
		/* ar[i] = -1.0f + 2.0f / (1.0f + expf(-2.0f*ar[i])); */
}

static void exponential( unsigned int n, float *ar )
{
	for( unsigned int i = 0; i < n; i++ )
		ar[i] = expf(ar[i]);
}

static void softplus( unsigned int n, float *ar )
{
	for( unsigned int i = 0; i < n; i++ )
		ar[i] = logf( expf(ar[i]) + 1.0f ) ;
}

static void softsign( unsigned int n, float *ar )
{
	for( unsigned int i = 0; i < n; i++ )
		ar[i] = ar[i] / (fabsf(ar[i]) + 1.0f ) ;
}

static void hard_sigmoid( unsigned int n, float *ar )
{
	for( unsigned int i = 0; i < n; i++ )
		ar[i] = ar[i] < -2.5f ? 0.0f :
                ar[i] >  2.5f ? 1.0f :
                0.2f * ar[i] + 0.5f ;
}

#if defined(TRAINING_FEATURES)

static void softplus_derivative    ( unsigned int n, const float *activation, float *ar )
{
    for( unsigned int i=0; i < n; i++ ){
        float x =  expf( activation[i] );
        ar[i] *= (x - 1.0f) / x; 
    }
}

static void softsign_derivative    ( unsigned int n, const float *activation, float *ar )
{
    /* Can this be simplified? */
    for( unsigned int i=0; i < n; i++ ){
        float x = activation[i] / (1.0f - fabsf(activation[i]));
        ar[i] *= 1.0f / ((1.0f + fabsf(x)) * (1.0f + fabsf(x)));
    }
}

static void hard_sigmoid_derivative( unsigned int n, const float *activation, float *ar )
{
    for( unsigned int i=0; i < n; i++ )
        ar[i] *= activation[i] <= 0.0f ? 0.0f :
           activation[i] >= 1.0f ? 0.0f : 0.2f ;
}
static void exponential_derivative ( unsigned int n, const float *activation, float *ar )
{
    for( unsigned int i=0; i < n; i++ )
        ar[i] *= activation[i];
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
static void linear_derivative      ( unsigned int n, const float *activation, float *ar )
{
    /* This function is intentionally empty. */
}

static void softmax_derivative     ( unsigned int n, const float *activation, float *ar )
{
    /* This function is intentionally empty. */
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static void relu_derivative        ( unsigned int n, const float *activation, float *ar )
{
    for( unsigned int i=0; i < n; i++ )
        ar[i] *= activation[i] <= 0.0f ? 0.0f : 1.0f ;
}

static void sigmoid_derivative     ( unsigned int n, const float *activation, float *ar )
{
    for( unsigned int i=0; i < n; i++ )
        ar[i] *= activation[i]*(1.0f-activation[i]);
}

static void tanh_act_derivative    ( unsigned int n, const float *activation, float *ar )
{
    for( unsigned int i=0; i < n; i++ )
        ar[i] *= 1.0f-activation[i]*activation[i];

}
#endif
