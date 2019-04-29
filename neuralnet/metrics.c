#include "metrics.h"

#include <stdint.h> 
#include <string.h>
#include <math.h>

typedef float (*metric_func)(unsigned int n, const float *y_pred, const float *y_real );

static float mean_squared_error            ( unsigned int n, const float *y_pred, const float *y_real );
static float mean_absolute_error           ( unsigned int n, const float *y_pred, const float *y_real );
static float mean_absolute_percentage_error( unsigned int n, const float *y_pred, const float *y_real );
static float binary_crossentropy           ( unsigned int n, const float *y_pred, const float *y_real );
static float categorical_crossentropy      ( unsigned int n, const float *y_pred, const float *y_real );

static float epsilon = 1.0e-7f;

metric_func get_metric_func( const char * name ){
	return
		!strcmp( name, "mean_squared_error") ? mean_squared_error :
		!strcmp( name, "mse")                ? mean_squared_error :


		!strcmp( name, "mean_absolute_error") ? mean_absolute_error :
		!strcmp( name, "mae")                 ? mean_absolute_error :

		!strcmp( name, "mean_absolute_percentage_error") ? mean_absolute_percentage_error :
		!strcmp( name, "mape")                           ? mean_absolute_percentage_error :


		!strcmp( name, "categorical_crossentropy")       ? categorical_crossentropy :
		!strcmp( name, "binary_crossentropy")            ? binary_crossentropy :

		NULL;
}

const char * get_metric_name( metric_func ptr ){
	return
		ptr == mean_squared_error             ? "mean_squared_error" :
		ptr == mean_absolute_error            ? "mean_absolute_error" :
		ptr == mean_absolute_percentage_error ? "mean_absolute_percentage_error" :
		ptr == binary_crossentropy            ? "binary_crossentropy" :
		ptr == categorical_crossentropy       ? "categorical_crossentropy" :
		"(unknown)";
}

/* 
   These are the real metric functions that will return a float. These
   should *NOT* be used as loss functions in the backpropagation.

   Here is a trick to get the correspnding metric function based on the
   loss function:

   metric_func metric = get_metric_func( get_loss_name( nn->loss ) );

*/

/* FIXME: These are now really plain and no SIMD yet. */
static float mean_squared_error( unsigned int n, const float *y_pred, const float *y_real)
{
    float err = 0.0f;
    for( unsigned int i = 0; i < n; i++ )
        err += (y_pred[i] - y_real[i] ) * (y_pred[i] - y_real[i]);

    return err / (float) n;
}

static float mean_absolute_error(unsigned int n, const float *y_pred, const float *y_real)
{
    float err = 0.0f;
    for( unsigned int i = 0; i < n; i++ )
        err += fabsf( y_real[i] - y_pred[i] );
    return err / (float) n;
}

static float mean_absolute_percentage_error(unsigned int n, const float *y_pred, const float *y_real)
{
    float diff = 0.0f;
    for( unsigned int i = 0; i < n; i++ )
        diff += fabsf( (y_real[i] - y_pred[i]) / fmaxf( y_real[i], epsilon ));

    return 100.0f * diff / (float) n;
}

/* These two functions can probably be joind in some way. fix this later. */
static float binary_crossentropy(unsigned int n, const float *y_pred, const float *y_real)
{
    float clipped_y_pred[n];
    for( unsigned int i = 0; i < n; i++ ){
        clipped_y_pred[i] = fmaxf( y_pred[i], epsilon );
        clipped_y_pred[i] = fminf( clipped_y_pred[i], 1.0f - epsilon );
    }

    float err = 0.0f;
    for( unsigned int i = 0; i < n; i++ )
        err += y_real[i] * logf( clipped_y_pred[i] ) + (1.0f - y_real[i]) * logf( 1.0f - clipped_y_pred[i]);

    return -err / (float) n ;
}

static float categorical_crossentropy(unsigned int n, const float *y_pred, const float *y_real)
{
    float clipped_y_pred[n];
    for( unsigned int i = 0; i < n; i++ ){
        clipped_y_pred[i] = fmaxf( y_pred[i], epsilon );
        clipped_y_pred[i] = fminf( clipped_y_pred[i], 1.0f - epsilon );
    }

    float err = 0.0f;
    for( unsigned int i = 0; i < n; i++ )
        err += y_real[i] * logf( clipped_y_pred[i] );

    return -err / (float) n ;
}
