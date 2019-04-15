#include "activation.h"
#include "exponential.h"

#include <stdint.h> 
#include <immintrin.h> 
#include <string.h>
#include <math.h>

static void logistic_gnubg  ( unsigned int hidden, float *ar );
static void logistic_plain  ( unsigned int hidden, float *ar );
static void rectifier_vec   ( unsigned int hidden, float *ar );
static void rectifier_plain ( unsigned int hidden, float *ar );
static void tanh_libc       ( unsigned int hidden, float *ar );
static void tanh_vec        ( unsigned int hidden, float *ar );
static void softmax( unsigned int hidden, float *ar );

activation_func get_activation_func( const char * name ){
	return
		!strcmp( name, "logistic_gnubg") ? logistic_gnubg :
		!strcmp( name, "logistic_plain") ? logistic_plain :
		!strcmp( name, "rectifier_vec") ? rectifier_vec :
		!strcmp( name, "rectifier_plain") ? rectifier_plain :
		!strcmp( name, "tanh_libc") ? tanh_libc:
		!strcmp( name, "tanh_vec") ? tanh_vec:
		!strcmp( name, "softmax") ? softmax:
		NULL;
}

const char * get_activation_name( activation_func ptr ){
	return
		ptr == logistic_gnubg ? "logistic_gnubg" :
		ptr == logistic_plain ? "logistic_plain" :
		ptr == rectifier_vec ? "rectifier_vec" :
		ptr == rectifier_plain ? "rectifier_plain" :
		ptr == tanh_libc ? "tanh_libc" :
		ptr == tanh_vec ? "tanh_vec" :
		ptr == softmax ? "softmax" :
		"(unknown)";
}
/* FIXME: There is actually no implementations for systems w/o AVX...  =:-o */

/* AVX implementations */
#define _PS_CONST8(Name, Val)											\
  static const __m256 _ps_8_##Name __attribute__((aligned(16))) = { Val, Val, Val, Val, Val, Val, Val, Val }

_PS_CONST8( ones, 1.0f );
_PS_CONST8( twos, 2.0f );
_PS_CONST8( tens, 10.0f );
_PS_CONST8( maxvals, EXP_MAX_VALUE );
_PS_CONST8( minusones, -1.0f );

static const int32_t __attribute__ ((aligned(16))) __abs_mask8[8] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF,
                                                                     0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};

/* Discuss: should this function be moved to exponential.h ?
 * Or maybe I should have one file for each simd technology? */
static inline __m256 exp_positive_avx( __m256 xin )
{
	xin = _mm256_min_ps( xin, _ps_8_maxvals );
	xin = _mm256_mul_ps( xin, _ps_8_tens );

#if defined(__AVX2__)
	__m256i i = _mm256_cvtps_epi32( xin );
	__m256 ex = _mm256_i32gather_ps( e, i, 4 );
	xin = _mm256_sub_ps( xin, _mm256_cvtepi32_ps( i ) );
#else  /* AVX but not AVX2 */
	union { __m256i i; int32_t elem[8]; } i;
	union { __m256 ex;   float elem[8]; } ex;
	i.i = _mm256_cvtps_epi32( xin );
#if 0
    for ( int j = 0; j < 8; j++ )
        ex.elem[j] = e[i.elem[j]];
#endif
	ex.elem[0] = e[i.elem[0]];
	ex.elem[1] = e[i.elem[1]];
	ex.elem[2] = e[i.elem[2]];
	ex.elem[3] = e[i.elem[3]];
	ex.elem[4] = e[i.elem[4]];
	ex.elem[5] = e[i.elem[5]];
	ex.elem[6] = e[i.elem[6]];
	ex.elem[7] = e[i.elem[7]];
	xin = _mm256_sub_ps( xin, _mm256_cvtepi32_ps( i.i ) );
#endif
	xin = _mm256_add_ps( xin, _ps_8_tens );
#if defined(__AVX2__)
	return _mm256_mul_ps( xin, ex );
#else
	return _mm256_mul_ps( xin, ex.ex );
#endif
}

/* FIXME: Maybe something that can be unified. These two functions are really sharing a lot. */
static inline __m256 tanhv_avx( __m256 xin )
{
	__m256 mask = _mm256_cmp_ps( xin, _mm256_setzero_ps(), _CMP_LT_OS );
	xin = _mm256_and_ps (xin , *(__m256*) __abs_mask8 ); /* Abs. value by clearing signbit */

	xin = _mm256_mul_ps( xin, _ps_8_twos );
	xin = exp_positive_avx( xin );
	xin = _mm256_div_ps( _mm256_sub_ps( xin, _ps_8_ones ), _mm256_add_ps( xin, _ps_8_ones ) );

	return _mm256_or_ps( _mm256_andnot_ps(  mask, xin ) , _mm256_and_ps ( mask , _mm256_mul_ps( _ps_8_minusones, xin )));
}

static inline __m256 logisticv_avx( __m256 xin )
{
	__m256 mask = _mm256_cmp_ps( xin, _mm256_setzero_ps(), _CMP_LT_OS );
	xin = _mm256_and_ps (xin , *(__m256*) __abs_mask8 ); /* Abs. value by clearing signbit */

	xin = exp_positive_avx( xin );
	xin = _mm256_rcp_ps( _mm256_add_ps( xin, _ps_8_ones ));

	return _mm256_or_ps( _mm256_and_ps(  mask, xin ) , _mm256_andnot_ps ( mask , _mm256_sub_ps( _ps_8_ones, xin )));
}

static void logistic_gnubg( unsigned int hidden, float *ar )
{
	unsigned int count = hidden >> 3;
	__m256 betavec = _mm256_set1_ps( 0.1f );
	float *par;
	for ( par = ar ; count; count--, par += 8){
		__m256 vec0 = _mm256_load_ps( par );
		vec0 = _mm256_mul_ps( vec0, betavec );
		vec0 = logisticv_avx( vec0 );
		_mm256_store_ps( par, vec0 );
	}
}

static void tanh_vec( unsigned int hidden, float *ar )
{
	unsigned int count = hidden >> 3;
	for ( float *par = ar; count; count--, par += 8 ){
		__m256 vec0 = _mm256_load_ps( par );
		vec0 = tanhv_avx( vec0 );
		_mm256_store_ps( par, vec0 );
	}
}

/* for reference */
static void tanh_libc( unsigned int hidden, float *ar )
{
	for ( unsigned int i = 0; i < hidden; i++ )
		ar[i] = tanhf( ar[i] );
} 

/* FIXME: Rewrite this */
static inline float logistic(float const xin)
{
/*    const float maxv = e[EXP_MAX_INDEX] * 10.0f;  */
	if( xin > 0.0f ) { 
		if( xin < EXP_MAX_VALUE ) {
			const float x1 = 10.0f * xin;
			const int i = (int)x1;

			return 1.0f / (1.0f + e[i] * ((10 - i) + x1));
		} else
            return 0.0f;
			/* return 1.0f / (maxv + 1.0f); */
	} else {
		if( xin > -EXP_MAX_VALUE ) {
			const float x1 = -10.0f * xin;
			const int i = (int)x1;
			return 1.0f - 1.0f / (1.0f + e[i] * ((10 - i) + x1));
		} else 
            return 1.0f;
			/* return maxv / (maxv + 1.0f); */
	}
}

static void logistic_plain( unsigned int hidden, float *ar )
{
	for( unsigned int i = 0; i < hidden ; i++ )
		ar[ i ] = logistic( -ar[i] );
}

static inline float lookup_exp( float const xin )
{
	const float maxv = e[EXP_MAX_INDEX] * 10.0f;  
	if( xin > 0.0f ) { 
		if( xin < EXP_MAX_VALUE ) {
			const float x1 = 10.0f * xin;
			const int i = (int)x1;

			return e[i] * ((10 - i) + x1);
		} else
			return maxv;
	} else {
		if( xin > -EXP_MAX_VALUE ) {
			const float x1 = -10.0f * xin;
			const int i = (int)x1;
			return 1.0f / (e[i] * ((10 - i) + x1));
		} else 
			return 1.0f / maxv;
	}
}

static void softmax( unsigned int hidden, float *ar )
{
	float sum = 0.0f;
	for ( unsigned int j = 0 ; j < hidden; j++ ){
		ar[j] = lookup_exp( ar[j] );
		sum += ar[j];
	}
	for ( unsigned int j = 0 ; j < hidden; j++ ){
		ar[j] /= sum;
	}
}

static void rectifier_vec( unsigned int hidden, float *ar )
{
	unsigned int count = hidden >> 3;
	for ( float *par = ar; count; count--, par += 8 ){
		__m256 vec0 = _mm256_load_ps( par );
		vec0 = _mm256_max_ps( _mm256_setzero_ps(), vec0 );
		_mm256_store_ps( par, vec0 );
	}
}

static void rectifier_plain( unsigned int hidden, float *ar )
{
	for( unsigned int i = 0; i < hidden; i++ )
		ar[i] = fmaxf(0.0f, ar[i]);
}
