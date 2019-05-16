#ifndef __MATRIX_MULTIPLY_H__
#define __MATRIX_MULTIPLY_H__
void matrix_vector_multiply( int m, int n, const float *weight, const float *y, float *out );
void vector_matrix_multiply( int n, int m, const float *weight, const float *bias, const float *input, float *y );
void vector_vector_outer   ( int n_rows, int n_cols, const float *x, const float *y, float *matrix );
#endif /* __MATRIX_MULTIPLY_H__ */
