#include "vector_operations.h"
#include <immintrin.h>

/**
 * @brief Add vectors a and b,  a = a + b 
 *
 * @param n Length of vector
 * @param a An array
 * @param b Another array
 */
/* This is actually the same as saxpy -- y = ax + y, but with a = 1.0 */
void vector_accumulate( const int n, float *a, const float *b )
{
    int i = 0;
    float *a_ptr = a;
    const float *b_ptr = b;
#ifdef __AVX__
    for ( ; i <= ((n)-8); i += 8, a_ptr += 8, b_ptr += 8 )
        _mm256_store_ps(a_ptr, _mm256_add_ps(_mm256_load_ps(a_ptr), _mm256_load_ps(b_ptr)));
#endif
    for (; i < n; i++ )
        *a_ptr++ += *b_ptr++; 
}

/**
 * @brief Scale a vector. v = scalar * v
 *
 * @param n Length of vector
 * @param v A vector array to be scaled
 * @param scalar A scalar value
 */
void vector_scale( const int n, float *v, const float scalar )
{
    int i = 0;
    float *v_ptr = v;
#ifdef __AVX__
    __m256 v_scale = _mm256_set1_ps(scalar);
    for ( ; i <= ((n)-8); i += 8, v_ptr += 8)
        _mm256_store_ps(v_ptr, _mm256_mul_ps(_mm256_load_ps(v_ptr), v_scale));
#endif
    for( ; i < n; i++ )
        *v_ptr++ *= scalar;
}

/**
 * @brief Scale a vector by division. v = scalar / v
 *
 * @param n Length of vector
 * @param v A vector array to be scaled
 * @param scalar A scalar value
 */
void vector_divide_by_scalar( const int n, float *v, const float scalar )
{
    int i = 0;
    float *v_ptr = v;
#ifdef __AVX__
    __m256 v_scale = _mm256_set1_ps(scalar);
    for ( ; i <= ((n)-8); i += 8, v_ptr += 8)
        _mm256_store_ps(v_ptr, _mm256_div_ps(_mm256_load_ps(v_ptr), v_scale));
#endif
    for( ; i < n; i++ )
        *v_ptr++ /= scalar;
}

/**
 * @brief Implements a saxpy operation. y = alpha * x + y.
 *
 * This function implements a saxpy operation pretty much like any BLAS implementation,
 * however this does not follow the BLAS standard. It is not threaded.
 *
 * @param n Length of the vector
 * @param a The vector (y)
 * @param alpha the scalar value
 * @param b The vector to be scaled and added.
 *
 * @warning This function asserts that the vectors are aligned. If they are not,
 * your code may crash. Make sure your code is aligned properly.
 */ 
void vector_saxpy( const int n, float *a, const float alpha, const float *b )
{
    int i = 0;
    float *a_ptr = a;
    const float *b_ptr = b;
#ifdef __AVX__
    __m256 v_scale = _mm256_set1_ps(alpha);
    for ( ; i <= ((n)-8); i += 8, a_ptr += 8, b_ptr += 8 )
        _mm256_store_ps(a_ptr, _mm256_add_ps(_mm256_load_ps(a_ptr), _mm256_mul_ps( v_scale,  _mm256_load_ps(b_ptr))));
#endif
    for (; i < n; i++ )
        *a_ptr++ += alpha * *b_ptr++; 
}

/**
 * @brief Implements a saxpby operation. y = alpha * x + beta * y.
 *
 * This function implements a saxpby operation pretty much like any BLAS implementation,
 * however this does not follow the BLAS standard when it comes to argument ordering.
 * The functions is not multi-threaded.
 *
 * @param n Length of the vector
 * @param a The vector (y)
 * @param alpha the scalar value
 * @param b The vector to be scaled and added.
 * @param beta The scalar for (y).
 *
 * @warning This function asserts that the vectors are aligned. If they are not,
 * your code may crash. Make sure your code is aligned properly.
 *
 * @warning There is no increment (stride), so your vectors must be sequential in memory
 */ 
void vector_saxpby( const int n, float *a, const float alpha, const float *b, const float beta )
{
    int i = 0;
    float *a_ptr = a;
    const float *b_ptr = b;
#ifdef __AVX__
    __m256 v_scale = _mm256_set1_ps(alpha);
    __m256 b_scale = _mm256_set1_ps(beta);
    for ( ; i <= ((n)-8); i += 8, a_ptr += 8, b_ptr += 8 )
        _mm256_store_ps(a_ptr,
                _mm256_add_ps( 
                    _mm256_mul_ps(_mm256_load_ps(a_ptr), b_scale),
                    _mm256_mul_ps( v_scale,  _mm256_load_ps(b_ptr))
                )
            );
#endif
    for (; i < n; i++, a_ptr++ )
        *a_ptr = beta * *a_ptr + alpha * *b_ptr++; 
}

/**
 * @brief squares each element in a vector: y = x*x (elementwise)
 *
 * @param n Length of the vector
 * @param y The vector containing the qsuared values.
 * @param x The input vector.
 */
void vector_square_elements ( const int n, float *y, const float *x )
{
    int i = 0;
    float *y_ptr = y;
    const float *x_ptr = x;
#ifdef __AVX__
    for ( ; i <= ((n)-8); i += 8, y_ptr += 8, x_ptr += 8 ){
        __m256 xvec = _mm256_load_ps( x_ptr );
        _mm256_store_ps(y_ptr, _mm256_mul_ps( xvec, xvec ));
    }
#endif
    for (; i < n; i++ ){
        const float xval = x[i];
        *y_ptr++ = xval * xval; 
    }
}
