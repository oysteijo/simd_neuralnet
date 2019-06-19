#ifndef __VECTOR_OPERATIONS_H__
#define __VECTOR_OPERATIONS_H__

/* Note. These functions are made for operating on parameter vectors, however, they are
   general enough to do any vector. The only thing to keep in mind is that these
   functions asserts that the input vectors are aligned */
void vector_accumulate      ( const int n, float *a, const float *b );
void vector_scale           ( const int n, float *v, const float scalar );
void vector_divide_by_scalar( const int n, float *v, const float scalar );
void vector_saxpy           ( const int n, float *y, const float alpha, const float *x );
void vector_saxpby          ( const int n, float *y, const float alpha, const float *x, const float beta );
void vector_square_elements ( const int n, float *y, const float *x );
#endif /* __VECTOR_OPERATIONS_H__ */
