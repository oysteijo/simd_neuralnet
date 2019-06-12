#include "vector_operations.h"
#include <immintrin.h>

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

