#ifndef __MATRIX_MULTIPLY_H__
#define __MATRIX_MULTIPLY_H__
void matrix_multiply_general( int n, int m, const float *weight, const float *bias, const float *input, float *y );
void matrix_multiply_output( int m, int n, const float *weight, const float *bias, const float *y, float *out );
#endif /* __MATRIX_MULTIPLY_H__ */
