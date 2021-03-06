/* Copyright -- Øystein Johansen 2007 */
/* $Id: $ */
/* FIXME - This files needs a cleanup */
#ifndef __SIMD_H__
#define __SIMD_H__

#include <stdlib.h>
#include <stdint.h>

#if defined(__SSE__) /* FIXME or other SIMD technologies */

/* FIXME find right header to include based on Compiler (need _mm_malloc()) */
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#elif defined( __GNUC__)
#include <mm_malloc.h>
#endif

#if defined(__AVX__)
#define ALIGN_SIZE 32
#else
#define ALIGN_SIZE 16
#endif

#if defined(_MSC_VER)
#define SIMD_ALIGN(D) __declspec(align(ALIGN_SIZE)) D
#else
#define SIMD_ALIGN(D) D __attribute__ ((aligned(ALIGN_SIZE)))
#endif

#define simd_aligned(ar) (!(((int)ar) % ALIGN_SIZE))

#define is_aligned(POINTER) \
    (((uintptr_t)(const void *)(POINTER)) % (ALIGN_SIZE) == 0)

#else
#define SIMD_ALIGN(D) D
#define simd_aligned(ar) ar
#endif /* USE_SSE_VECTORIZE */

static inline float *
simd_malloc (size_t size)
{
#if defined(__SSE__)
    return (float *) _mm_malloc (size, ALIGN_SIZE);
#else
    return (float *) malloc (size);
#endif
}

static inline void
simd_free (float *ptr)
{
#if defined(__SSE__)
    _mm_free (ptr);
#else
    free (ptr);
#endif
}

#endif /* __SIMD_H__ */
