#include "loss.h"

#include <stdint.h> 
#include <string.h>
#include <math.h>

static void mean_squared_error            ( unsigned int n, const float *y_pred, const float *y_real, float *loss );
static void mean_absolute_error           ( unsigned int n, const float *y_pred, const float *y_real, float *loss );
static void mean_absolute_percentage_error( unsigned int n, const float *y_pred, const float *y_real, float *loss );
static void binary_crossentropy           ( unsigned int n, const float *y_pred, const float *y_real, float *loss );
static void categorical_crossentropy      ( unsigned int n, const float *y_pred, const float *y_real, float *loss );

loss_func get_loss_func( const char * name ){
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

const char * get_loss_name( loss_func ptr ){
	return
		ptr == mean_squared_error             ? "mean_squared_error" :
		ptr == mean_absolute_error            ? "mean_absolute_error" :
		ptr == mean_absolute_percentage_error ? "mean_absolute_percentage_error" :
		ptr == binary_crossentropy            ? "binary_crossentropy" :
		ptr == categorical_crossentropy       ? "categorical_crossentropy" :
		"(unknown)";
}

/* 
   Note to developers who want to implement their own loss function:
   There is no automatic (symbolic) derivation of a computation graph.
   The functions defined in this code is therefore the derivative
   w.r.t the output. However there are some logic in neuralnet_set_loss
   that modifies the activation output for matching output activations.
   The functions can therefore look a bit strange in some cases.

   If you don't understand the concept of "matching", I really recommend
   you to do the derivation by hand on paper.

   Also note that these functions does not calculate the scalar loss. These
   functions return "void" instead of float. If you really need the scalar
   loss value, you have to use the corresponding **metric function. These
   functions are ment for internal usage in the backpropagation.
*/

/* FIXME: These are now really plain and no SIMD yet. However, I doubt that these will rock
 * the boat if/when they were vectorized to SIMD. Put priority on vectorizing activations instead. */
static void mean_squared_error( unsigned int n, const float *y_pred, const float *y_real, float *loss )
{
    for( unsigned int i = 0; i < n; i++ )
        loss[i] = 2.0f * (y_pred[i] - y_real[i] ) / (float) n;
}

static void mean_absolute_error( unsigned int n, const float *y_pred, const float *y_real, float *loss )
{
    for( unsigned int i = 0; i < n; i++ )
        loss[i] = (y_pred[i] >= y_real[i] ? 1.0f : -1.0f) / (float) n;
}

static void mean_absolute_percentage_error( unsigned int n, const float *y_pred, const float *y_real, float *loss )
{
    for( unsigned int i = 0; i < n; i++ )
        loss[i] = (y_pred[i] >= y_real[i] ? 100.0f : -100.0f) / (fmax(y_real[i], 1e-7f) * (float) n);
}

static void categorical_crossentropy( unsigned int n, const float *y_pred, const float *y_real, float *loss )
{
    for( unsigned int i = 0; i < n; i++ )
        loss[i] = (y_pred[i] - y_real[i]);
}

#if 0
/* I'm not really satisfied with this code. I'm leaving this dead for now. Here are some issues:

   * if the target class is given as integers, it breaks the "interface" declaration. Of course the
     loss function declaration can have a void pointer as the target, but then I have to cast inside
     each and every function. Well...
   * I need changes in all the optimizers, I need to change the neuralnet code. I need change sevaral places.
   * I will need special handling every time this is used */
*/
static void sparse_categorical_crossentropy_int( unsigned int n, const float *y_pred, const int target_class, float *loss )
{
    assert( target_class < n );
    float arr[n];
    memset( arr, 0, n * sizeof(float));
    arr[target_class] = 1.0f;
    categorical_crossentropy( n, y_pred, arr, loss );
}

static void sparse_categorical_crossentropy( unsigned int n, const float *y_pred, const float *y_real, float *loss )
{
    int target_class = (int) round(*y_real);
    sparse_categorical_crossentropy_int( n, y_pred, target_class, loss );
}
#endif

static void binary_crossentropy( unsigned int n, const float *y_pred, const float *y_real, float *loss )
{
    for( unsigned int i = 0; i < n; i++ )
        loss[i] = (y_pred[i] - y_real[i]) / (float) n;
}

