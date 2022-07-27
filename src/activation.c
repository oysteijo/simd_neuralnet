#include "activation.h"

#include <string.h>
#include <math.h>

#include <stdio.h>
#include <errno.h>
#ifdef __AVX__
#include <immintrin.h>
#endif

#include <dlfcn.h>

/*
Just thinking out here.... This code needs a cleanup. It started as a implementation
from 'octopus', but at this stage we rather want things to be general than superfast.

Browsing the documentation from an other neural network library (Keras), I see that
the provided activation functions are:

sigmoid
softmax
elu      (Uses a parameter)
selu     (Uses a parameter or two?)
softplus
softsign
relu
tanh
hard_sigmoid
exponential
linear

Advanced activation functions
-----------------------------
LeakyReLU (Uses a parameter)
PReLU     (Uses a trainable parameter!)

OK! Since C does not support closures to generate functions, I guess I will have to wait with activations
using a parameter. (I have to hide the parameter inside a struct and return a pointer to this struct. The
struct will contain the parameter and the function pointer. Let's wait with this.... )

This leaves us with:
sigmoid, softmax, softplus, softsign, relu, tanh, hard_sigmoid, exponential, linear
*/

static void softplus    ( const int n, float *ar );
static void softsign    ( const int n, float *ar );
static void hard_sigmoid( const int n, float *ar );
static void exponential ( const int n, float *ar );
static void linear      ( const int n, float *ar );
static void relu        ( const int n, float *ar );
static void softmax     ( const int n, float *ar );
static void sigmoid     ( const int n, float *ar );
/* Argh! "tanh" is already defined in math.h  (C99). We give this a different name */
static void tanh_act    ( const int n, float *ar );

#ifndef PREDICTION_ONLY
static void softplus_derivative    ( const int n, const float *activation, float *ar );
static void softsign_derivative    ( const int n, const float *activation, float *ar );
static void hard_sigmoid_derivative( const int n, const float *activation, float *ar );
static void exponential_derivative ( const int n, const float *activation, float *ar );
static void linear_derivative      ( const int n, const float *activation, float *ar );
static void relu_derivative        ( const int n, const float *activation, float *ar );
static void softmax_derivative     ( const int n, const float *activation, float *ar );
static void sigmoid_derivative     ( const int n, const float *activation, float *ar );
static void tanh_act_derivative    ( const int n, const float *activation, float *ar );
#endif
#define __USE_DYNAMIC_LOAD__ 1
#if __USE_DYNAMIC_LOAD__ == 1
#include <dlfcn.h>
void cleanup_dynamic_symbols();

typedef struct _activation_record_t activation_record_t;
struct _activation_record_t {
    char *activation_name;
    void *handle;
    activation_func func_ptr;
    activation_derivative deriv_ptr;
    activation_record_t *next;
};

static activation_record_t *records = NULL;  /* A linked list of open activation funcs. */

static activation_func get_activation_func_dynamic( const char *name )
{
    atexit(cleanup_dynamic_symbols);
    /* strdup() is not ansi c .... */
    size_t len = strlen( name ) + 1;
    char *file_and_symbol_name = malloc( len );
    if ( !file_and_symbol_name ){
        perror("malloc");
        return NULL;
    }
    memcpy( file_and_symbol_name, name, len);

    char *at = strchr( file_and_symbol_name, '@' );
    if( !at ){
        fprintf(stderr, "Trying to load dynamic function, but name does not contain any '@'\n"
                "Format should read '<symbol>@<libraryfile>'\n");
        free( file_and_symbol_name );
        return NULL;
    }

    *at = '\0';
    char *library_file = at + 1;
    char *symbol = file_and_symbol_name;

    printf("library file: %s\n", library_file);  
    printf("symbol (function): %s\n", symbol);

    void *handle = dlopen( library_file, RTLD_NOW );
    if ( !handle ){
        fprintf( stderr, "Cannot open dynamic library '%s'. \n"
                "Set LD_LIBRARY_PATH \n"
                "or install it in a directory where dynamic linker finds it,\n"
                "or link the library with -Wl,-rpath  linker option\n", library_file);
        free( file_and_symbol_name );
        return NULL;
    }
    activation_func retfunc = dlsym( handle, symbol );
    char derivative_symbol[256];
    sprintf(derivative_symbol, "%s_derivative", symbol );
    free( file_and_symbol_name );
    char * error_message = dlerror();
    if( error_message )
    {
        fprintf( stderr, "dlsym(): %s\n", error_message );
        dlclose( handle );
        return NULL;
    }

    activation_record_t *rec = malloc( sizeof(activation_record_t) );
    if ( !rec ){
        return retfunc;
    }

    /* Need to strdup again ... (Or maybe I can reuse if I re-insert the @ ?) */
    rec->handle = handle;
    rec->activation_name = malloc( len );
    if ( !rec->activation_name ){
        perror("malloc");
        return retfunc;
    }
    memcpy( rec->activation_name, name, len);

    rec->deriv_ptr = dlsym( handle, derivative_symbol);
    error_message = dlerror();
    if( error_message )
    {
        fprintf( stderr, "dlsym(): %s\n", error_message );
        fprintf( stderr, "This neural net will not be trainable and expect a segmentation fault if you try.\n");
        return retfunc;
    }
    rec->func_ptr = retfunc;
    rec->next = NULL;

    /* Find the backmost and append */
    if( !records )
        records = rec;
    else {
        activation_record_t *endptr = records;
        while(endptr->next)
            endptr = endptr->next;
        endptr->next = rec;
    }

    return retfunc;
    /* the dl will be open to the bitter end. there is no explicit call to dlclose() when everything works fine.
       I really hope that doesn't matter. */
}

#if 0
int debug_linked_list()
{
    if( !records )
        return 0;
    int count = 0;
    activation_record_t *iter = records;
    do {
        printf( "Name                  : %s\n", iter->activation_name ); 
        printf( "Address               : %p\n", iter->func_ptr ); 
        printf( "Address of derivative : %p\n", iter->deriv_ptr ); 
        iter = iter->next;
        count++;
    } while ( iter );
    return count;
}
#endif

activation_derivative get_activation_derivative_dynamic( const activation_func ptr )
{
    if( !records )
        return NULL;
    activation_record_t *iter = records;
    do {
        if( iter->func_ptr == ptr ) return iter->deriv_ptr; 
        iter = iter->next;
    } while ( iter );
    return NULL;
}

void cleanup_dynamic_symbols()
{
    /* FIXME: Clean debug output */
    printf("Cleaning up...");
    if( !records ){
        printf("No dynamic symbols found. Have a nice day!\n");
        return;
    }
    activation_record_t *iter = records;
    do {
        dlclose( iter->handle );
        free( iter->activation_name );
        activation_record_t *to_be_set_free = iter;
        iter = iter->next;
        free( to_be_set_free );
    } while ( iter );
    printf("All clean now!\n");
}
#endif /* __USE_DYNAMIC_LOAD__ */

#define CHECK_ACTIVATION_NAME(func) \
        !strcmp( name, #func) ? func :

activation_func get_activation_func( const char * name )
{
    return
        CHECK_ACTIVATION_NAME(softplus)
        CHECK_ACTIVATION_NAME(softsign)
        CHECK_ACTIVATION_NAME(hard_sigmoid)
        CHECK_ACTIVATION_NAME(exponential)
        CHECK_ACTIVATION_NAME(linear)
        CHECK_ACTIVATION_NAME(relu)
        CHECK_ACTIVATION_NAME(softmax)
        CHECK_ACTIVATION_NAME(sigmoid)
        !strcmp( name, "tanh") ? tanh_act :
        get_activation_func_dynamic( name );
}

#define CHECK_ACTIVATION_PTR(func) \
        ptr == func ? #func :

const char * get_activation_name( activation_func ptr ){
    return
        CHECK_ACTIVATION_PTR(softplus)
        CHECK_ACTIVATION_PTR(softsign)
        CHECK_ACTIVATION_PTR(hard_sigmoid)
        CHECK_ACTIVATION_PTR(exponential)
        CHECK_ACTIVATION_PTR(linear)
        CHECK_ACTIVATION_PTR(relu)
        CHECK_ACTIVATION_PTR(softmax)
        CHECK_ACTIVATION_PTR(sigmoid)
        ptr == tanh_act ? "tanh" :
        "(unknown)";
}
#undef CHECK_ACTIVATION_NAME
#undef CHECK_ACTIVATION_PTR

#ifndef PREDICTION_ONLY
#define CHECK_ACTIVATION_DERIV_PTR(func) \
        ptr == func ? func ## _derivative :

activation_derivative get_activation_derivative( activation_func ptr ){
    return
        CHECK_ACTIVATION_DERIV_PTR(softplus)
        CHECK_ACTIVATION_DERIV_PTR(softsign)
        CHECK_ACTIVATION_DERIV_PTR(hard_sigmoid)
        CHECK_ACTIVATION_DERIV_PTR(exponential)
        CHECK_ACTIVATION_DERIV_PTR(linear)
        CHECK_ACTIVATION_DERIV_PTR(relu)
        CHECK_ACTIVATION_DERIV_PTR(softmax)
        CHECK_ACTIVATION_DERIV_PTR(sigmoid)
        CHECK_ACTIVATION_DERIV_PTR(tanh_act)
        get_activation_derivative_dynamic( ptr );
}
#undef CHECK_ACTIVATION_DERIV_PTR
#endif

static void relu( const int n, float *y )
{
    int i = 0;
#ifdef __AVX__
    const __m256 zero = _mm256_set1_ps(0.0f);

    __m256 YMM0, YMM1;

    for (i = 0; i <= ((n)-16); i += 16) {
        YMM0 = _mm256_load_ps(y + i);
        YMM1 = _mm256_load_ps(y + i + 8);
        YMM0 = _mm256_max_ps(zero, YMM0);
        YMM1 = _mm256_max_ps(zero, YMM1);
        _mm256_store_ps( y + i, YMM0 );
        _mm256_store_ps( y + i + 8, YMM1 );
    }
#endif
    for( ; i < n; i++ )
        y[i] = fmaxf(0.0f, y[i]);
}

#ifdef __AVX2__
#if defined(__GNUC__)
# define ALIGN32_BEG __attribute__((aligned(32)))
#elif defined(_WIN32)
# define ALIGN32_BEG __declspec(align(32))
#endif

#define _PS256_CONST(Name, Val)                                            \
  static const ALIGN32_BEG float _ps256_##Name[8] = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PI32_CONST256(Name, Val)                                            \
  static const ALIGN32_BEG int _pi32_256_##Name[8] = { Val, Val, Val, Val, Val, Val, Val, Val }

_PS256_CONST(1  , 1.0f);
_PS256_CONST(0p5, 0.5f);

_PS256_CONST(exp_hi,        88.3762626647949f);
_PS256_CONST(exp_lo,        -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

_PI32_CONST256(0x7f, 0x7f);

static inline __m256 exp256_ps(__m256 x) {
    __m256 tmp = _mm256_setzero_ps(), fx;
    __m256i imm0;
    __m256 one = *(__m256*)_ps256_1;

    x = _mm256_min_ps(x, *(__m256*)_ps256_exp_hi);
    x = _mm256_max_ps(x, *(__m256*)_ps256_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_LOG2EF);
    fx = _mm256_add_ps(fx, *(__m256*)_ps256_0p5);

    /* how to perform a floorf with SSE: just below */
    //imm0 = _mm256_cvttps_epi32(fx);
    //tmp  = _mm256_cvtepi32_ps(imm0);

    tmp = _mm256_floor_ps(fx);

    /* if greater, substract 1 */
    //__m256 mask = _mm256_cmpgt_ps(tmp, fx);
    __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    tmp = _mm256_mul_ps(fx, *(__m256*)_ps256_cephes_exp_C1);
    __m256 z = _mm256_mul_ps(fx, *(__m256*)_ps256_cephes_exp_C2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    z = _mm256_mul_ps(x,x);

    __m256 y = *(__m256*)_ps256_cephes_exp_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_exp_p5);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm256_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm256_add_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
    imm0 = _mm256_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}
#endif 

#ifdef __SSE3__
static inline float hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}
#endif

#ifdef __AVX2__
/* Does horizontal sum - see:
   https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-float-vector-sum-on-x86 */
static inline float hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow  = _mm_add_ps(vlow, vhigh);     // add the low 128
    return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
    // (no wasted instructions, and all of them are the 4B minimum)
}
#endif

static void softmax( const int n, float *ar )
{
    /* There is an excellent article on how to do it here:
     * https://arxiv.org/pdf/2001.04438.pdf
     * The Two-Pass Softmax Algorithm - Marat Dukhan and Artsiom Ablavatski */
    float sum = 0.0f;
    float maxval = ar[0];
    int j = 1;
    
    /* This follows the three-pass with re-loading (Algorithm 2 in the article).
       However I have not found a fast SIMD way to find the maximum element.
       If you get to a time critical training of a classification problem with
       many classes, it may pay off implement the two-pass algorithm */
    for (; j < n; j++ )
        if( ar[j] > maxval ) maxval = ar[j];

    j = 0;
#ifdef __AVX2__
    /* I'm intentionally only using one register to make the vectorization work for "8 or more"-class
       classification problems. If using two registers, I will lose all vectorization for classification
       problems with less than 16 classes. This is of course a trade off, and if you ever do a classification
       problem with 16 or more classes, you could consider re-writing. */
    __m256 max_v = _mm256_set1_ps( maxval );
    __m256 sum_v = _mm256_set1_ps( 0.0f );
    for (; j <= ((n)-8); j += 8) {
        __m256 YMM0 = _mm256_load_ps(ar + j);
        YMM0 = _mm256_sub_ps( YMM0, max_v );
        YMM0 = exp256_ps(YMM0);
        _mm256_store_ps( ar + j, YMM0 );
        sum_v = _mm256_add_ps( sum_v, YMM0 );
    }
    sum += hsum256_ps_avx( sum_v );
#endif
    for (; j < n; j++ ){
        ar[j] = expf( ar[j] - maxval );
        sum += ar[j];
    }
    j = 0;
#ifdef __AVX__
    __m256 sum4 = _mm256_set1_ps( sum );
    for(; j <= ((n)-8); j+= 8 ) {
        __m256 YMM0 = _mm256_load_ps( ar + j );
        _mm256_store_ps( ar + j, _mm256_div_ps( YMM0, sum4 ) );
    }
#endif
    for (; j < n; j++ ){
        ar[j] /= sum;
    }
}

static void sigmoid( const int n, float *y )
{
    int i = 0;
#ifdef __AVX2__
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_set1_ps(0.0f);

    __m256 YMM0, YMM1, YMM2, YMM3;

    for (i = 0; i <= ((n)-16); i += 16) {
        YMM0 = _mm256_load_ps(y + i);
        YMM1 = _mm256_load_ps(y + i + 8);
        YMM0 = _mm256_sub_ps(zero, YMM0);
        YMM1 = _mm256_sub_ps(zero, YMM1);
        YMM2 = _mm256_add_ps(one, exp256_ps(YMM0));
        YMM3 = _mm256_add_ps(one, exp256_ps(YMM1));
        YMM2 = _mm256_div_ps(one, YMM2);
        YMM3 = _mm256_div_ps(one, YMM3);
        _mm256_store_ps(y + i, YMM2);
        _mm256_store_ps(y + i + 8, YMM3);
    }
#endif  /* __AVX2__ */
    for (; i < (n); i++) {
        y[i] = 1.0f / (1.0f + expf(-y[i]));
    }
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
static void linear( const int n, float *ar )
{
    /* Do nothing */
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static void tanh_act( const int n, float *y )
{
    int i = 0;
#ifdef __AVX2__
    const __m256 one      = _mm256_set1_ps( 1.0f);
    const __m256 neg_one  = _mm256_set1_ps(-1.0f);
    const __m256 two      = _mm256_set1_ps( 2.0f);
    const __m256 neg_two  = _mm256_set1_ps(-2.0f);

    __m256 YMM0, YMM1, YMM2, YMM3;

    for (i = 0; i <= ((n)-16); i += 16) {
        YMM0 = _mm256_load_ps(y + i);
        YMM1 = _mm256_load_ps(y + i + 8);
        YMM0 = _mm256_mul_ps(neg_two, YMM0);
        YMM1 = _mm256_mul_ps(neg_two, YMM1);
        YMM2 = _mm256_add_ps(one, exp256_ps(YMM0));
        YMM3 = _mm256_add_ps(one, exp256_ps(YMM1));
        YMM2 = _mm256_div_ps(two, YMM2);
        YMM3 = _mm256_div_ps(two, YMM3);
        YMM2 = _mm256_add_ps(neg_one, YMM2);
        YMM3 = _mm256_add_ps(neg_one, YMM3);
        _mm256_store_ps(y + i, YMM2);
        _mm256_store_ps(y + i + 8, YMM3);
    }
#endif  /* __AVX2__ */
    for( ; i < n; i++ )
        y[i] = tanhf(y[i]);
}

/* These are really seldom used activation functions. We can vectorize these when needed */
static void exponential( const int n, float *ar )
{
    for( int i = 0; i < n; i++ )
        ar[i] = expf(ar[i]);
}

static void softplus( const int n, float *ar )
{
    for( int i = 0; i < n; i++ )
        ar[i] = logf( expf(ar[i]) + 1.0f ) ;
}

static void softsign( const int n, float *ar )
{
    for( int i = 0; i < n; i++ )
        ar[i] = ar[i] / (fabsf(ar[i]) + 1.0f ) ;
}

static void hard_sigmoid( const int n, float *ar )
{
    for( int i = 0; i < n; i++ )
        ar[i] = ar[i] < -2.5f ? 0.0f :
            ar[i] >  2.5f ? 1.0f :
            0.2f * ar[i] + 0.5f ;
}

#ifndef PREDICTION_ONLY
static void softplus_derivative    ( const int n, const float *activation, float *ar )
{
    for( int i=0; i < n; i++ ){
        float x =  expf( activation[i] );
        ar[i] *= (x - 1.0f) / x; 
    }
}

static void softsign_derivative    ( const int n, const float *activation, float *ar )
{
    /* Can this be simplified? */
    for( int i=0; i < n; i++ ){
        float x = activation[i] / (1.0f - fabsf(activation[i]));
        ar[i] *= 1.0f / ((1.0f + fabsf(x)) * (1.0f + fabsf(x)));
    }
}

static void hard_sigmoid_derivative( const int n, const float *activation, float *ar )
{
    int i = 0;
    for( ; i < n; i++ )
        ar[i] *= activation[i] <= 0.0f ? 0.0f :
           activation[i] >= 1.0f ? 0.0f : 0.2f ;
}
static void exponential_derivative ( const int n, const float *activation, float *ar )
{
    for( int i=0; i < n; i++ )
        ar[i] *= activation[i];
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
static void linear_derivative      ( const int n, const float *activation, float *ar )
{
    /* This function is intentionally empty. */
}

static void softmax_derivative     ( const int n, const float *activation, float *ar )
{
    /* This function is intentionally empty. */
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static void relu_derivative        ( const int n, const float *activation, float *ar )
{
    int i = 0;
#ifdef __AVX__
    const __m256 zero = _mm256_setzero_ps();
    const __m256 ones = _mm256_set1_ps(1.0f);

    __m256 YMM0, YMM1, YMM2, YMM3;

    for (i = 0; i <= ((n)-16); i += 16) {
        YMM0 = _mm256_loadu_ps(activation + i);
        YMM1 = _mm256_loadu_ps(activation + i + 8);
        
        YMM0 = _mm256_cmp_ps(YMM0, zero, _CMP_LE_OS);
        YMM1 = _mm256_cmp_ps(YMM1, zero, _CMP_LE_OS);

        YMM0 = _mm256_blendv_ps( ones, zero, YMM0);
        YMM1 = _mm256_blendv_ps( ones, zero, YMM1);

        YMM2 = _mm256_loadu_ps(ar + i);
        YMM3 = _mm256_loadu_ps(ar + i + 8);

        _mm256_storeu_ps( ar + i,     _mm256_mul_ps( YMM0, YMM2) );
        _mm256_storeu_ps( ar + i + 8, _mm256_mul_ps( YMM1, YMM3) );
    }
#endif /* __AVX__ */
    for(; i < n; i++ )
        ar[i] *= activation[i] <= 0.0f ? 0.0f : 1.0f ;
}

static void sigmoid_derivative     ( const int n, const float *activation, float *ar )
{
    /* is the memory aligned? */
    int i = 0;
#ifdef __AVX__
    const __m256 ones = _mm256_set1_ps(1.0f);

    __m256 YMM0, YMM1, YMM2, YMM3;

    for (i = 0; i <= ((n)-16); i += 16) {
        YMM0 = _mm256_loadu_ps(activation + i);
        YMM1 = _mm256_loadu_ps(activation + i + 8);

        YMM2 = _mm256_sub_ps( ones, YMM0 );
        YMM3 = _mm256_sub_ps( ones, YMM1 );

        YMM0 = _mm256_mul_ps(YMM0, YMM2);
        YMM1 = _mm256_mul_ps(YMM1, YMM3);

        YMM2 = _mm256_loadu_ps(ar + i);
        YMM3 = _mm256_loadu_ps(ar + i + 8);

        _mm256_storeu_ps( ar + i,     _mm256_mul_ps( YMM0, YMM2) );
        _mm256_storeu_ps( ar + i + 8, _mm256_mul_ps( YMM1, YMM3) );
    }
#endif /* __AVX__ */
    for(; i < n; i++ )
        ar[i] *= activation[i]*(1.0f-activation[i]);
}

static void tanh_act_derivative    ( const int n, const float *activation, float *ar )
{
    int i = 0;
#ifdef __AVX__
    const __m256 ones = _mm256_set1_ps(1.0f);

    __m256 YMM0, YMM1, YMM2, YMM3;

    for (i = 0; i <= ((n)-16); i += 16) {
        YMM0 = _mm256_loadu_ps(activation + i);
        YMM1 = _mm256_loadu_ps(activation + i + 8);

        YMM0 = _mm256_mul_ps(YMM0, YMM0);
        YMM1 = _mm256_mul_ps(YMM1, YMM1);

        YMM0 = _mm256_sub_ps(ones, YMM0);
        YMM1 = _mm256_sub_ps(ones, YMM1);

        YMM2 = _mm256_loadu_ps(ar + i);
        YMM3 = _mm256_loadu_ps(ar + i + 8);

        _mm256_storeu_ps( ar + i,     _mm256_mul_ps( YMM0, YMM2) );
        _mm256_storeu_ps( ar + i + 8, _mm256_mul_ps( YMM1, YMM3) );
    }
#endif /* __AVX__ */
    for( ; i < n; i++ )
        ar[i] *= 1.0f-activation[i]*activation[i];

}
#endif /* PREDICTION_ONLY */
