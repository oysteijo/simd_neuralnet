#include "matrix_multiply.h"
#include <immintrin.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

void vector_accumulate_unaligned( const int n, float *y, const float *b )
{
#ifdef USE_CBLAS
    cblas_saxpy( n, 1.0f, b, 1, y, 1 );
#else
    float *y_ptr = y;
    const float *b_ptr = b;

    int i = 0;
#ifdef __AVX__
    for(; i <= ((n)-8); i += 8, y_ptr += 8, b_ptr += 8 )
        _mm256_storeu_ps(y_ptr, _mm256_add_ps(_mm256_loadu_ps(y_ptr), _mm256_loadu_ps(b_ptr) ));
#endif
    for (; i < n; i++ )
        *y_ptr++ += *b_ptr++;
#endif /* USE_CBLAS */
}

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
#ifdef USE_CBLAS
    cblas_sgemv( CblasRowMajor, CblasNoTrans,
            n_rows, n_cols, 1.0f, matrix, n_cols, v, 1, 0.0f, y, 1 );
#else
    const float *m_ptr = matrix;
    for( int i = 0; i < n_rows; i++ ){
        const float *v_ptr = v;
        int j = 0;
#ifdef __AVX__
		__m256 sum = _mm256_setzero_ps ();
		for (; j <= ((n_cols)-8); j += 8, m_ptr += 8, v_ptr += 8) /* Check if faster: unroll w prefetch */
   #if defined(__AVX2__)
			sum = _mm256_fmadd_ps( _mm256_load_ps(v_ptr), _mm256_load_ps(m_ptr), sum);
   #else
			sum = _mm256_add_ps (sum, _mm256_mul_ps(_mm256_load_ps(v_ptr), _mm256_load_ps(m_ptr)));
   #endif
		y[i] = horizontalsum_avx( sum );
#endif
        for(; j < n_cols; j++ )
            y[i] += *v_ptr++ * *m_ptr++;
    }
#endif
}


void vector_vector_outer( int n_rows, int n_cols, const float *x, const float *y, float *matrix )
{
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
#ifdef __AVX__
            __m256 scale = _mm256_set1_ps( a );
            for( ; j <= ((n_cols)-8); j += 8, y_ptr += 8, matrix_ptr += 8) {
                /* Unforuneately this has to go unaligned. Can that be fixed? */
                _mm256_storeu_ps( matrix_ptr, _mm256_mul_ps( scale, _mm256_loadu_ps( y_ptr )) );
            }
#endif  /* __AVX__ */
            for( ; j < n_cols; j++ )
                *matrix_ptr++ = a * *y_ptr++;
        }
    }
#endif /* USE_CBLAS */
}

void vector_matrix_multiply( int n, int m, const float *weight, const float *bias, const float *input, float *y )
{
    /* Use cblas_sgemv on this? */
    const float *bias_ptr = bias;
	float *y_ptr = y; 
    int i = 0;
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
		const float *weight_ptr = weight + ( i * m );
		if (inp) {
			float  *y_ptr = y;
			if (inp == 1.0f){
                int j = 0;
#ifdef __AVX__
				for (; j <= ((m)-8) ; j += 8, y_ptr += 8, weight_ptr += 8) 
					_mm256_store_ps(y_ptr, _mm256_add_ps (_mm256_load_ps(y_ptr), _mm256_load_ps( weight_ptr )));
#endif /*  __AVX__ */
                for( ; j < m; j++ )
                    *y_ptr++ += *weight_ptr++;
            }

			else {
                int j = 0;
#ifdef __AVX__
				__m256 scalevec = _mm256_set1_ps(inp);
				for (; j < ((m)-8) ; j += 8, y_ptr += 8, weight_ptr += 8){
   #if defined(__AVX2__)
					_mm256_store_ps(y_ptr, _mm256_fmadd_ps( _mm256_load_ps(weight_ptr), scalevec, _mm256_load_ps(y_ptr)));
   #else
					_mm256_store_ps(y_ptr, _mm256_add_ps(_mm256_load_ps(y_ptr), _mm256_mul_ps(_mm256_load_ps(weight_ptr), scalevec)));
   #endif  
                }
#endif
                for(; j < m; j++ )
                    *y_ptr++ += inp * *weight_ptr++;
			}
		}
	}
}

