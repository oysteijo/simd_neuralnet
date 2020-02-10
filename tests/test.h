/* test.h */
/* A simple include file to do some testing */

#include <stddef.h>
#include <math.h>

#ifndef __TEST_H__
#define __TEST_H__
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define OK   KGRN "(OK)" KNRM
#define FAIL KRED "FAIL" KNRM

#define CHECK_INT_EQUALS_MSG(a,b,msg)               \
    test_count++;                                   \
    fprintf(stderr, "%-72s: %s\n", msg , a==b ? OK : FAIL); \
    if( a != b ) fail_count++;

#define CHECK_NOT_NULL_MSG(a,msg)               \
    test_count++;                                   \
    fprintf(stderr, "%-72s: %s\n", msg , a!=NULL ? OK : FAIL); \
    if( a == NULL ) fail_count++;

#define CHECK_STR_EQUALS_MSG(a,b,msg)               \
    test_count++;                                   \
    fprintf(stderr, "%-72s: %s\n", msg , strcmp(a,b)==0 ? OK : FAIL); \
    if( strcmp(a,b)==0 ) fail_count++;

#define CHECK_FLOAT_EQUALS_MSG(a,b,eps,msg) \
    test_count++;  \
    fprintf(stderr, "%-72s: %s\n", msg , fabsf( (a)-(b) ) <= eps ? OK : FAIL); \
    if( fabsf( (a)-(b) ) > eps ) fail_count++;
    
#define CHECK_CONDITION_MSG(cond,msg) \
    test_count++;  \
    fprintf(stderr, "%-72s: %s\n", msg , (cond) ? OK : FAIL); \
    if( !(cond) ) fail_count++;
    
/* I've prefixed these functions with 'test_' as they are only supposed to be
 * used for testing. the implemetations are naiive and probably very slow. */
static inline float test_calculate_mean( size_t n, float *values )
{
    float sum = 0.0f;
    float *ptr = values;
    size_t counter = n;
    while( counter-- ){
        sum += *ptr++;
    }
    return sum / (float) n;
}

static inline float test_calculate_stddev( size_t n, float *values )
{
    if( n < 2 ) return 0.0f;

    float    sum = 0.0f;
    float sq_sum = 0.0f;

    for ( unsigned int i = 0; i < n; i++ ){
        const float val = values[i];
        sum    += val;
        sq_sum += val*val;
    }

    float mean = sum / n;
    float var  = sq_sum / n - mean*mean;

    return sqrtf( var );
}

static inline float test_calculate_max( size_t n, float *values )
{
    float maxval = *values;
    float *ptr = values;
    size_t counter = n;
    while( counter-- ){
        if( *ptr > maxval ) maxval = *ptr;;
        ptr++;
    }
    return maxval;
}

static inline float test_calculate_min( size_t n, float *values )
{
    float minval = *values;
    float *ptr = values;
    size_t counter = n;
    while( counter-- ){
        if( *ptr < minval ) minval = *ptr;;
        ptr++;
    }
    return minval;
}

#define print_test_summary(n_tests,fail_tests) \
    fprintf(stderr, "------------------------------------\n"); \
    fprintf(stderr, " Summary of '%s'.\n", __FILE__); \
    fprintf(stderr, "------------------------------------\n"); \
    fprintf(stderr, " Total tests run   :  %14d\n", n_tests); \
    fprintf(stderr, " Total tests failed:  %14d\n", fail_tests); \
    fprintf(stderr, "------------------------------------\n"); 
    
#endif  /* __TEST_H__ */

