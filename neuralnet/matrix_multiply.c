#include "matrix_multiply.h"
#include <immintrin.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

/* This is quite hard to make much more effective with SIMD instructions. I can give it a shot in some ifdefs, but
   I'll leave an alternative to cblas as well, as I think that maight me the best option for this functions. */
void matrix_vector_multiply( int n_rows, int n_cols, const float *matrix, const float *v, float *y )
{
#ifdef USE_CBLAS
    cblas_sgemv( CblasRowMajor, CblasNoTrans,
            n_rows, n_cols, 1.0f, matrix, n_cols, v, 1, 0.0f, y, 1 );
#else
    const float *m_ptr = matrix;
    for( int i = 0; i < n_rows; i++ ){
        const float *v_ptr = v;
        for( int j = 0; j < n_cols; j++ )
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
				for (; j < ((m)-8) ; j += 8, y_ptr += 8, weight_ptr += 8) 
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

