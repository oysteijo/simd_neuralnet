/* matrix_operations.c - Øystein Schønning-Johansen 2013 - 2023 */
/* 
 vim: ts=4 sw=4 softtabstop=4 expandtab 
*/
#include "matrix_operations.h"
#include "simd.h"
#include <assert.h>

#ifdef __AVX__ 
#include <immintrin.h>
#endif

#ifdef USE_CBLAS
#include <cblas.h>
#include <string.h>
#endif

#ifdef __AVX__  
static inline float horizontalsum_avx( __m256 x )
{
    float sumAVX = 0.0f;
    __m256 hsum = _mm256_hadd_ps(x, x);
    hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
    _mm_store_ss(&sumAVX, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );
    return sumAVX;
}
#endif

/* This is quite hard to make much more effective with SIMD instructions. I can give it a shot in some ifdefs.
   (UPDATE: I just did, and it saves a few fractions! Yeah!), but I'll leave an alternative to cblas as well,
   as I think that might be the best option for this function. */
void matrix_vector_multiply( int n_rows, int n_cols, const float *matrix, const float *v, float *y )
{
    /*
    assert( is_aligned( matrix ));  // Seems to be aligned
    assert( is_aligned( v ));       // Not aligned correctly (add padding to the gradient?)
    assert( is_aligned( y ));       // Nope! Not properly aligned either - which is not so strange actually. But does it matter for y? 
    */
#ifdef USE_CBLAS
    cblas_sgemv( CblasRowMajor, CblasNoTrans,
            n_rows, n_cols, 1.0f, matrix, n_cols, v, 1, 0.0f, y, 1 );
#else
    const float *m_ptr = matrix;
    for( int i = 0; i < n_rows; i++ ){
        const float *v_ptr = v;
        int j = 0;

#ifdef __AVX512F__
        __m512 sums = _mm512_setzero_ps ();
        for (; j <= ((n_cols)-16); j += 16, m_ptr += 16, v_ptr += 16){ /* Check if faster: unroll w prefetch */
#if defined(__FMA__)
            sums = _mm512_fmadd_ps( _mm512_loadu_ps(v_ptr), _mm512_load_ps(m_ptr), sums);
#else
            sums = _mm512_add_ps (sums, _mm512_mul_ps(_mm512_loadu_ps(v_ptr), _mm512_load_ps(m_ptr)));
#endif
        }
        y[i] = _mm512_reduce_add_ps( sums );
#endif
#ifdef __AVX__
        __m256 sum = _mm256_setzero_ps ();
        for (; j <= ((n_cols)-8); j += 8, m_ptr += 8, v_ptr += 8){ /* Check if faster: unroll w prefetch */
#if defined(__FMA__)
            sum = _mm256_fmadd_ps( _mm256_loadu_ps(v_ptr), _mm256_load_ps(m_ptr), sum);
#else
            sum = _mm256_add_ps (sum, _mm256_mul_ps(_mm256_loadu_ps(v_ptr), _mm256_load_ps(m_ptr)));
#endif
        }
        y[i] += horizontalsum_avx( sum );
#endif
        for(; j < n_cols; j++ )
            y[i] += *v_ptr++ * *m_ptr++;
    }
#endif /* USE_CBLAS */
}

void vector_vector_outer( int n_rows, int n_cols, const float *x, const float *y, float *matrix )
{
    /* printf("Is y vector aligned? %s\n", is_aligned( y ) ? "Yes" : "Nope" );
    It is not always aligned! I guess I have to align the different elements of the gradient for that. */
#ifdef USE_CBLAS
    cblas_sger(CblasRowMajor, n_rows, n_cols, 1.0, x, 1, y, 1, matrix, n_cols);
#else
    /* At entry the matrix should be initialized to zeros. This is also requiered by cblas_sger() so there is no loss */
    for ( int i = 0; i < n_rows; i++ ){
        const float a = x[i];
        if( a ) {
            int j = 0;
            float *matrix_ptr = matrix + ( i * n_cols );
            const float *y_ptr = y;
#ifdef __AVX512F__
            __m512 scale512 = _mm512_set1_ps( a );
            for( ; j <= ((n_cols)-16); j += 16, y_ptr += 16, matrix_ptr += 16) {
                _mm512_storeu_ps( matrix_ptr, _mm512_mul_ps( scale512, _mm512_loadu_ps( y_ptr )) );
            }
#endif  /* __AVX512F__ */
#ifdef __AVX__
            __m256 scale256 = _mm256_set1_ps( a );
            for( ; j <= ((n_cols)-8); j += 8, y_ptr += 8, matrix_ptr += 8) {
                /* Unforuneately this has to go unaligned. Can that be fixed? */
                _mm256_storeu_ps( matrix_ptr, _mm256_mul_ps( scale256, _mm256_loadu_ps( y_ptr )) );
            }
#endif  /* __AVX__ */
            for( ; j < n_cols; j++ )
                *matrix_ptr++ = a * *y_ptr++;
        }
    }
#endif /* USE_CBLAS */
}

/* Discuss: We can maybe check if if m is a multiple of ALIGN_SIZE at entry, and
 * in case it is, we can load_ps and store_ps with aligned instead of unaligned.
 * However, I'm not sure how much it will improve the performance.  */
void vector_matrix_multiply( int n, int m, const float *weight, const float *bias, const float *input, float *y )
{
#ifdef USE_CBLAS
    memcpy( y, bias, m * sizeof(float));
    cblas_sgemv( CblasRowMajor, CblasTrans,
            n, m, 1.0f, weight, m, input, 1, 1.0f, y, 1 );
#else
    /*
    assert( is_aligned( weight ));
    assert( is_aligned( bias ));
    assert( is_aligned( y ));
    */
    const float *bias_ptr = bias;
    float *y_ptr = y; 
    int i = 0;

#ifdef __AVX512F__
    for (; i <= ((m)-16) ; i += 16, bias_ptr +=16, y_ptr +=16 ){
        _mm512_store_ps( y_ptr, _mm512_load_ps( bias_ptr ));
    }
#endif
#ifdef __AVX__
    for (; i <= ((m)-8) ; i += 8, bias_ptr +=8, y_ptr +=8 ){
        _mm256_store_ps( y_ptr, _mm256_load_ps( bias_ptr ));
    }
#endif
    for( ; i < m; i++ ){
        *y_ptr++ = *bias_ptr++;
    }

    for (int i = 0; i < n; i++) {
        float const inp = input[i];
        const float *weight_ptr = weight + ( i * m );  /* Argh! if m is not a multiple of ALIGN_SIZE, the pointer will be unaligned! :-( */
        if (inp) {
            float  *y_ptr = y;  /* same goes for this */
            if (inp == 1.0f){
                int j = 0;
#ifdef __AVX512F__
                for (; j <= ((m)-16) ; j += 16, y_ptr += 16, weight_ptr += 16) 
                    _mm512_store_ps(y_ptr, _mm512_add_ps (_mm512_load_ps(y_ptr), _mm512_loadu_ps( weight_ptr )));
#endif /*  __AVX512F__ */
#ifdef __AVX__
                for (; j <= ((m)-8) ; j += 8, y_ptr += 8, weight_ptr += 8) 
                    _mm256_store_ps(y_ptr, _mm256_add_ps (_mm256_load_ps(y_ptr), _mm256_loadu_ps( weight_ptr )));
#endif /*  __AVX__ */
                for( ; j < m; j++ )
                    *y_ptr++ += *weight_ptr++;
            } else {
                int j = 0;
#ifdef __AVX512F__
                for (; j < ((m)-16) ; j += 16, y_ptr += 16, weight_ptr += 16){
#if defined(__FMA__)
                    _mm512_store_ps(y_ptr, _mm512_fmadd_ps( _mm512_loadu_ps(weight_ptr), _mm512_set1_ps(inp), _mm512_load_ps(y_ptr)));
#else
                    _mm512_store_ps(y_ptr, _mm512_add_ps(_mm512_load_ps(y_ptr), _mm512_mul_ps(_mm512_loadu_ps(weight_ptr), _mm512_set1_ps(inp))));
#endif  
                }
#endif
#ifdef __AVX__
                __m256 scalevec = _mm256_set1_ps(inp);
                for (; j < ((m)-8) ; j += 8, y_ptr += 8, weight_ptr += 8){
#if defined(__FMA__)
                    _mm256_store_ps(y_ptr, _mm256_fmadd_ps( _mm256_loadu_ps(weight_ptr), scalevec, _mm256_load_ps(y_ptr)));
#else
                    _mm256_store_ps(y_ptr, _mm256_add_ps(_mm256_load_ps(y_ptr), _mm256_mul_ps(_mm256_loadu_ps(weight_ptr), scalevec)));
#endif  
                }
#endif
                for(; j < m; j++ )
                    *y_ptr++ += inp * *weight_ptr++;
            }
        }
    }
#endif /* USE_CBLAS */
}

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
#ifdef USE_CBLAS
    cblas_saxpy( n, 1.0f, b, 1, a, 1 );
#else
    int i = 0;
    float *a_ptr = a;
    const float *b_ptr = b;
#ifdef __AVX512F__
    for ( ; i <= ((n)-16); i += 16, a_ptr += 16, b_ptr += 16 )
        _mm512_store_ps(a_ptr, _mm512_add_ps(_mm512_load_ps(a_ptr), _mm512_load_ps(b_ptr)));
#endif
#ifdef __AVX__
    for ( ; i <= ((n)-8); i += 8, a_ptr += 8, b_ptr += 8 )
        _mm256_store_ps(a_ptr, _mm256_add_ps(_mm256_load_ps(a_ptr), _mm256_load_ps(b_ptr)));
#endif
    for (; i < n; i++ )
        *a_ptr++ += *b_ptr++; 
#endif /* USE_CBLAS */
}

/**
 * @brief Add vectors a and b,  a = a + b 
 *
 * @param n Length of vector
 * @param a An array
 * @param b Another array
 */
 /* This is actually the same as above, but without the alignment requirement */
void vector_accumulate_unaligned( const int n, float *y, const float *b )
{
#ifdef USE_CBLAS
    cblas_saxpy( n, 1.0f, b, 1, y, 1 );
#else
    float *y_ptr = y;
    const float *b_ptr = b;

    int i = 0;
#ifdef __AVX512F__
    for(; i <= ((n)-16); i += 16, y_ptr += 16, b_ptr += 16 )
        _mm512_storeu_ps(y_ptr, _mm512_add_ps(_mm512_loadu_ps(y_ptr), _mm512_loadu_ps(b_ptr) ));
#endif
#ifdef __AVX__
    for(; i <= ((n)-8); i += 8, y_ptr += 8, b_ptr += 8 )
        _mm256_storeu_ps(y_ptr, _mm256_add_ps(_mm256_loadu_ps(y_ptr), _mm256_loadu_ps(b_ptr) ));
#endif
    for (; i < n; i++ )
        *y_ptr++ += *b_ptr++;
#endif /* USE_CBLAS */
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
#ifdef __AVX512F__
    for ( ; i <= ((n)-16); i += 16, v_ptr += 16)
        _mm512_store_ps(v_ptr, _mm512_mul_ps(_mm512_load_ps(v_ptr), _mm512_set1_ps(scalar)));
#endif
#ifdef __AVX__    
    for ( ; i <= ((n)-8); i += 8, v_ptr += 8)
        _mm256_store_ps(v_ptr, _mm256_mul_ps(_mm256_load_ps(v_ptr), _mm256_set1_ps(scalar)));
#endif
    for( ; i < n; i++ )
        *v_ptr++ *= scalar;
}

/**
 * @brief Scale a vector by division. v = v / scalar
 *
 * @param n Length of vector
 * @param v A vector array to be scaled
 * @param scalar A scalar value
 */
void vector_divide_by_scalar( const int n, float *v, const float scalar )
{
    int i = 0;
    float *v_ptr = v;
#ifdef __AVX512F__
    for ( ; i <= ((n)-16); i += 16, v_ptr += 16)
        _mm512_store_ps(v_ptr, _mm512_div_ps(_mm512_load_ps(v_ptr), _mm512_set1_ps(scalar)));
#endif
#ifdef __AVX__
    for ( ; i <= ((n)-8); i += 8, v_ptr += 8)
        _mm256_store_ps(v_ptr, _mm256_div_ps(_mm256_load_ps(v_ptr), _mm256_set1_ps(scalar)));
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
#ifdef USE_CBLAS
    cblas_saxpy( n, alpha, b, 1, a, 1 );
#else
    int i = 0;
    float *a_ptr = a;
    const float *b_ptr = b;
#ifdef __AVX512F__    
    for ( ; i <= ((n)-16); i += 16, a_ptr += 16, b_ptr += 16 )
        _mm512_store_ps(a_ptr, _mm512_add_ps(_mm512_load_ps(a_ptr), _mm512_mul_ps( _mm512_set1_ps(alpha),  _mm512_load_ps(b_ptr))));
#endif
#ifdef __AVX__
    for ( ; i <= ((n)-8); i += 8, a_ptr += 8, b_ptr += 8 )
        _mm256_store_ps(a_ptr, _mm256_add_ps(_mm256_load_ps(a_ptr), _mm256_mul_ps( _mm256_set1_ps(alpha),  _mm256_load_ps(b_ptr))));
#endif
    for (; i < n; i++ )
        *a_ptr++ += alpha * *b_ptr++; 
#endif /* USE_CBLAS */
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

#if USE_CBLAS // # Eh? is it not the blas standard?  
    cblas_saxpby( n, alpha, b, 1, beta, a, 1 );
#else
    int i = 0;
    float *a_ptr = a;
    const float *b_ptr = b;
#ifdef __AVX512F__
    __m512 v_scale_s = _mm512_set1_ps(alpha);
    __m512 b_scale_s = _mm512_set1_ps(beta);
    for ( ; i <= ((n)-16); i += 16, a_ptr += 16, b_ptr += 16 )
        _mm512_store_ps(a_ptr,
                _mm512_add_ps( 
                    _mm512_mul_ps(_mm512_load_ps(a_ptr), b_scale_s),
                    _mm512_mul_ps( v_scale_s,  _mm512_load_ps(b_ptr))
                )
            );
#endif

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
#endif /* USE_CBLAS */
}

/**
 * @brief squares each element in a vector: y = x*x (elementwise)
 *
 * @param n Length of the vector
 * @param y The vector containing the (output) squared values.
 * @param x The input vector.
 */
void vector_square_elements ( const int n, float *y, const float *x )
{
    int i = 0;
    float *y_ptr = y;

#ifdef __AVX__
    const float *x_ptr = x;
#endif

#ifdef __AVX512F__    
    for ( ; i <= ((n)-16); i += 16, y_ptr += 16, x_ptr += 16 ){
        __m512 xvec = _mm512_load_ps( x_ptr );
        _mm512_store_ps(y_ptr, _mm512_mul_ps( xvec, xvec ));
    }
#endif
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
