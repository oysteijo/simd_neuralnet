#include "matrix_multiply.h"
#include <immintrin.h>

void matrix_vector_multiply( int n_rows, int n_cols, const float *matrix, const float *v, float *y )
{
    const float *m_ptr = matrix;
    for( int i = 0; i < n_rows; i++ ){
        const float *v_ptr = v;
        for( int j = 0; j < n_cols; j++ )
            y[i] += *v_ptr++ * *m_ptr++;
    }
}

/**
  * This is plain vector dot matrix and is written to be used in backprop. It is not general */
void vector_matrix_multiply( int n_rows, int n_cols, const float *v, const float *matrix, float *y )
{
    for ( int i = 0; i < n_rows ; i++ ){
		float const v_value = v[i];
		const float *matrix_ptr = matrix + ( i * n_cols );
        float  *y_ptr = y;
        int j = 0;
#ifdef __AVX8__
        __m256 scale = _mm256_set1_ps( v_value );
        for( ; j <= ((n_cols)-8); j += 8, y_ptr += 8, matrix_ptr += 8) {
            /* Unforuneately this has to go unaligned. Can that be fixed? */
#if defined(__AVX2__)
            _mm256_storeu_ps(y_ptr, _mm256_fmadd_ps( _mm256_loadu_ps(matrix_ptr), scale, _mm256_loadu_ps(y_ptr)));
#else
            _mm256_storeu_ps(y_ptr, _mm256_add_ps(_mm256_loadu_ps(y_ptr), _mm256_mul_ps(_mm256_loadu_ps(matrix_ptr), scale)));
#endif
        }
#endif
        for ( ; j < n_cols; j++ )
            *y_ptr++ += v_value * *matrix_ptr++;
    }
}

void vector_vector_outer( int n_rows, int n_cols, const float *x, const float *y, float *matrix )
{
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
#endif
            for( ; j < n_cols; j++ )
                *matrix_ptr++ = a * *y_ptr++;
        }
    }
}

#if defined(__AVX__)
static inline float horizontalsum_avx( __m256 x )
{
	float sumAVX = 0.0f;
	__m256 hsum = _mm256_hadd_ps(x, x);
	hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
	_mm_store_ss(&sumAVX, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );
	return sumAVX;
}

void matrix_multiply_general( int n, int m, const float *weight, const float *bias, const float *input, float *y )
{
	const unsigned int count = m >> 3;
	const float *bias_ptr = bias;
	float *y_ptr = y; 
	for (int i = count; i ; i--, bias_ptr +=8, y_ptr +=8 )
		_mm256_store_ps( y_ptr, _mm256_load_ps( bias_ptr ));

	for (int i = 0; i < n; i++) {
		float const inp = input[i];
		const float *weight_ptr = weight + ( i * m );
		if (inp) {
			float  *y_ptr = y;
			if (inp == 1.0f)
				for (int j = count; j; j--, y_ptr += 8, weight_ptr += 8) 
					_mm256_store_ps(y_ptr, _mm256_add_ps (_mm256_load_ps(y_ptr), _mm256_load_ps( weight_ptr )));
			else {
				__m256 scalevec = _mm256_set1_ps(inp);
				for (int j = count; j; j--, y_ptr += 8, weight_ptr += 8)
#if defined(__AVX2__)
					_mm256_store_ps(y_ptr, _mm256_fmadd_ps( _mm256_load_ps(weight_ptr), scalevec, _mm256_load_ps(y_ptr)));
#else
					_mm256_store_ps(y_ptr, _mm256_add_ps(_mm256_load_ps(y_ptr), _mm256_mul_ps(_mm256_load_ps(weight_ptr), scalevec)));
#endif
			}
		}
	}
    /* Do the rest. If the user has done his/her homework, this should not be necesarry */
    unsigned int reminding = m & 0x7;
    if( !reminding ) return;

    y_ptr = y + (count*8);
    bias_ptr = bias + (count*8);

    for (int j = reminding; j; j--)
        *y_ptr++ = *bias_ptr++;

    for (int i = 0; i < n; i++) {
        float const inp = input[i];
        const float *weight_ptr = weight + ( i * m );
        if (inp) {
            float  *y_ptr = y + (count*8);
            if (inp == 1.0f)
                for (int j = reminding; j; j--)
                    *y_ptr++ += *weight_ptr++;
            else {
                for (int j = reminding; j; j--)
                    *y_ptr++ += inp * *weight_ptr++;
            }
        }
    }
}

void matrix_multiply_output( int m, int n, const float *weight, const float *bias, const float *y, float *out )
{
	const unsigned int count = m >> 3;
	for (int i = 0; i < n; i++) {
		const float  *y_ptr = y;
		const float  *weight_ptr = weight + ( i * m );
		__m256 sum = _mm256_setzero_ps ();
		for (int j = count; j; j--, weight_ptr += 8, y_ptr += 8) /* Check if faster: unroll w prefetch */
#if defined(__AVX2__)
			sum = _mm256_fmadd_ps( _mm256_load_ps(y_ptr), _mm256_load_ps(weight_ptr), sum);
#else
			sum = _mm256_add_ps (sum, _mm256_mul_ps(_mm256_load_ps(y_ptr), _mm256_load_ps(weight_ptr)));
#endif
		out[i] = horizontalsum_avx( sum ) + bias[i];
	}
}
#elif defined (__SSE__) /* End of __AVX__ */

static inline float horizontalsum_sse( __m128 in )
{
	float r;
	__m128 vec0 = _mm_shuffle_ps (in, in, _MM_SHUFFLE (2, 3, 0, 1));
	__m128 vec1 = _mm_add_ps (in, vec0);
	vec0 = _mm_shuffle_ps (vec1, vec1, _MM_SHUFFLE (1, 1, 3, 3));
	_mm_store_ss (&r, _mm_add_ps (vec1, vec0));
	return r;
}

static inline float horizontalsum_sse3( __m128 in )
{
	float r;
	in = _mm_hadd_ps ( in , in );
	in = _mm_hadd_ps ( in , in );
	_mm_store_ss (&r, in);
	return r;
}

#if defined(__SSE3__)
#define horizontalsum horizontalsum_sse3
#else
#define horizontalsum horizontalsum_sse
#endif

void matrix_multiply_general( int n, int m, const float *weight, const float *bias, const float *input, float *y )
{
	const unsigned int count = m >> 2;
	const float *bias_ptr = bias;
	float *y_ptr = y;

	for (int i = count; i ; i--, bias_ptr +=4, y_ptr +=4 )
		_mm_store_ps( y_ptr, _mm_load_ps( bias_ptr ));

	for (int i = 0; i < n; i++) {
		float const inp = input[i];
		const float *weight_ptr = weight + ( i * m );
		if (inp) {
			float  *y_ptr = y;
			if (inp == 1.0f)
				for (int j = count; j; j--, y_ptr += 4, weight_ptr += 4) 
					_mm_store_ps (y_ptr, _mm_add_ps (_mm_load_ps(y_ptr), _mm_load_ps( weight_ptr )));
			else {
				__m128 scalevec = _mm_set1_ps (inp);
				for (int j = count; j; j--, y_ptr += 4, weight_ptr += 4)
					_mm_store_ps(y_ptr, _mm_add_ps(_mm_load_ps(y_ptr), _mm_mul_ps(_mm_load_ps(weight_ptr), scalevec)));
			}
		}
	}
}

void matrix_multiply_output( int m, int n, const float *weight, const float *bias, const float *y, float *out )
{
	const unsigned int count = m >> 2;
	for (int i = 0; i < n; i++) {
		const float  *y_ptr = y;
		const float  *weight_ptr = weight + ( i * m );
		__m128 sum = _mm_setzero_ps ();
		for (int j = count; j; j--, weight_ptr += 4, y_ptr += 4) 
			sum = _mm_add_ps (sum, _mm_mul_ps(_mm_load_ps(y_ptr), _mm_load_ps(weight_ptr)));
		out[ i ] = horizontalsum( sum ) + nn->bias_o[ i ];
    }
}
#else /* End of __SSE__ */
/* Plain boring implementation, but some compilers manage to vectorize it. */
void matrix_multiply_general( int n, int m, const float *weight, const float *bias, const float *input, float *y )
{
	/* Calculate activity at hidden nodes */
	for (int i = 0; i < m; i++)
		y[i] = bias[i];

	for (int i = 0; i < n; i++) {
		float const inp = input[i];
		const float *weight_ptr = weight + ( i * m );

		if (inp) {
			float *y_ptr = y;
			if (inp == 1.0f)
				for (int j = m; j; j--)
					*y_ptr++ += *weight_ptr++;
			else
				for (int j = m; j; j--)
					*y_ptr++ += *weight_ptr++ * inp;
		}
	}
}

void matrix_multiply_output( int m, int n, const float *weight, const float *bias, const float *y, float *out )
{
	float *weight_ptr = weight;

	for (int i = 0; i < n; i++) {
		out[i] = bias[i];
		for ( int j = 0; j < m; j++)
			out[i] += y[j] * *weight_ptr++;
	}
}
#endif 
