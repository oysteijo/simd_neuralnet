#include "loss.h"
#include "exponential.h"

#include <stdint.h> 
#include <immintrin.h> 
#include <string.h>
#include <math.h>

typedef void (*loss_func)(const unsigned int n, const float *y_pred, const float *y_real, float *loss );

static void mean_squared_error            (const unsigned int n, const float *y_pred, const float *y_real, float *loss );
static void mean_absolute_error           (const unsigned int n, const float *y_pred, const float *y_real, float *loss );
static void mean_absolute_percentage_error(const unsigned int n, const float *y_pred, const float *y_real, float *loss );
static void crossentropy                  (const unsigned int n, const float *y_pred, const float *y_real, float *loss );

loss_func get_loss_func( const char * name ){
	return
		!strcmp( name, "mean_squared_error") ? mean_squared_error :
		!strcmp( name, "mse")                ? mean_squared_error :


		!strcmp( name, "mean_absolute_error") ? mean_absolute_error :
		!strcmp( name, "mae")                 ? mean_absolute_error :

		!strcmp( name, "mean_absolute_percentage_error") ? mean_absolute_percentage_error :
		!strcmp( name, "mape")                           ? mean_absolute_percentage_error :


		!strcmp( name, "categorical_crossentropy")       ? crossentropy :
		!strcmp( name, "binary_crossentropy")            ? crossentropy :
		!strcmp( name, "crossentropy")                   ? crossentropy :

		NULL;
}

const char * get_loss_name( loss_func ptr ){
	return
		ptr == mean_squared_error             ? "mean_squared_error" :
		ptr == mean_absolute_error            ? "mean_absolute_error" :
		ptr == mean_absolute_percentage_error ? "mean_absolute_percentage_error" :
		ptr == crossentropy                   ? "crossentropy" :
		"(unknown)";
}

/* 
   Note to developers who want to implement their own loss function:
   There is no automatic (symbolic) derivation of a comutation graph.
   The functions defined in this code is therefore the derivative
   w.r.t the output. However there are some logic in neuralnet_set_loss
   that modifies the activation output for matching output activations.
   The functions can therefor look a bit strange in some cases.

   If you don't understand the concept of "matching", I really recommend
   you to do the derivation by hand on paper.
*/

/* FIXME: These are now really plain and no SIMD yet. */
static void mean_squared_error(const unsigned int n, const float *y_pred, const float *y_real, float *loss )
{
    for( int i = 0; i < n; i++ )
        loss[i] = 2.0f * (y_pred[i] - y_real[i] ) / (float) n;
}

static void mean_absolute_error(const unsigned int n, const float *y_pred, const float *y_real, float *loss )
{
    for( int i = 0; i < n; i++ )
        loss[i] = y_pred[i] >= y_real[i] ? 1.0f : -1.0f;
}

static void mean_absolute_percentage_error(const unsigned int n, const float *y_pred, const float *y_real, float *loss )
{
    for( int i = 0; i < n; i++ )
        loss[i] = y_pred[i] >= y_real[i] ? 1.0f / (float) n : -1.0f / (float) n;
}

static void crossentropy(const unsigned int n, const float *y_pred, const float *y_real, float *loss )
{
    for( int i = 0; i < n; i++ )
        loss[i] = (y_pred[i] >= y_real[i]);
}
