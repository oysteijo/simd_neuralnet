/* matrix_operations.h - Øystein Schønning-Johansen 2019 - 2023 */
/*
  vim: ts=4 sw=4 softtabstop=4 expandtab 
 */
#ifndef __MATRIX_OPERATIONS_H__
#define __MATRIX_OPERATIONS_H__

/* These functions should only be used by optimizers and the neuralnet! */
/* Thay are changed continously, so use with care. */

void matrix_vector_multiply( int m, int n, const float *weight, const float *y, float *out );
void vector_matrix_multiply( int n, int m, const float *weight, const float *bias, const float *input, float *y );
void vector_vector_outer   ( int n_rows, int n_cols, const float *x, const float *y, float *matrix );

/* Note. These functions are made for operating on parameter vectors, however, they are
   general enough to do any vector. The only thing to keep in mind is that these
   functions asserts that the input vectors are aligned, except for the 
   obvious vector_accumulate_unaligned... */

void vector_accumulate          ( const int n, float *a, const float *b );
void vector_accumulate_unaligned( const int n, float *y, const float *b );
void vector_scale               ( const int n, float *v, const float scalar );
void vector_divide_by_scalar    ( const int n, float *v, const float scalar );
void vector_saxpy               ( const int n, float *y, const float alpha, const float *x );
void vector_saxpby              ( const int n, float *y, const float alpha, const float *x, const float beta );
void vector_square_elements     ( const int n, float *y, const float *x );
#endif /* __MATRIX_OPERATIONS_H__ */
