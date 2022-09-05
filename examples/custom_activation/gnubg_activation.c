#include <math.h>
#ifndef SIGMOID_SCALAR
#define SIGMOID_SCALAR 0.1
#endif

void scaled_sigmoid( const int n, float *y )
{
    for (int i = 0; i < n; i++) 
        y[i] = 1.0f / (1.0f + expf(-SIGMOID_SCALAR*y[i]));
}

void scaled_sigmoid_derivative( const int n, const float *activation, float *d )
{
    for(int i=0; i < n; i++ )
        d[i] *= SIGMOID_SCALAR*activation[i]*(1.0f-activation[i]);
}
