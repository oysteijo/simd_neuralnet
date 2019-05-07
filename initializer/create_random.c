#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "c_npy.h"

/* Returns a random float number from -1 to +1 with a uniform probability distribution. */
static float random_uniform()
{
    return (float) 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
}

static float random_normal()
{
    static float v, fac;
    static bool use_prev = false;
    float S, Z, u;
    if(use_prev)
        Z = v * fac;
    else {
        do {
            u = random_uniform();  /* from -1 to +1 */
            v = random_uniform();  /* from -1 to +1 */

            S = u*u + v*v;
        } while (S >= 1.0f);

        fac = sqrtf( -2.0f * logf(S) / S );
        Z = u * fac;
    }

    use_prev = !use_prev;
    return Z;
}

static void fill_data( unsigned int n, float (*dist)(), float scale, float *data )
{
    float *ptr = data;
    for ( unsigned int i = n; i--; )
        *ptr++ = dist() * scale;
}

static int create_and_save_random_weights( int n_layers, int sizes[], char *initializer[], const char *filename )
{
#if 0
    printf("n_layers: %d\n", n_layers );
    for ( int i = 0 ; i < n_layers+1 ; i++ )
        printf("size: %d\n", sizes[i] );

    for ( int i = 0 ; i < n_layers ; i++ )
        printf("init: %s\n", initializer[i] );
#endif
    cmatrix_t *array[n_layers*2 + 1]; /* bias and weight for each layer pluss a terminating NULL; */

    for ( int i = 0; i < n_layers ; i++ ){
        const int n_inp = sizes[i];
        const int n_out = sizes[i+1];

        /* weight */
        array[2*i] = calloc( 1, sizeof( cmatrix_t ));
        assert( array[2*i] );
        array[2*i]->data         = malloc( n_inp * n_out * sizeof(float));
        assert(array[2*i]->data);
        array[2*i]->shape[0]     = n_inp;
        array[2*i]->shape[1]     = n_out;
        array[2*i]->ndim         = 2;
        array[2*i]->endianness   = '<' ;  /* FIXME */
        array[2*i]->typechar     = 'f' ;
        array[2*i]->elem_size    = sizeof(float) ;
        array[2*i]->fortran_order= false ;

        if( strcmp( initializer[i], "xavier" ) == 0 )
            fill_data( n_inp * n_out, random_uniform, sqrtf(6.0f / (n_inp+n_out)), (float*) array[2*i]->data );
        else if( strcmp( initializer[i], "kaiming" ) == 0 )
            fill_data( n_inp * n_out, random_normal, sqrtf(2.0f/n_inp), (float*) array[2*i]->data );
        else {
            printf("Warning: Initializers not recognized at layer %d. normal distributed random values used.\n", i);
            fill_data( n_inp * n_out, random_normal, 1.0f, (float*) array[2*i]->data );
        }

        /* bias */
        array[2*i + 1] = calloc( 1, sizeof( cmatrix_t ));
        assert( array[2*i + 1] );
        array[2*i + 1]->data         = malloc( n_out * sizeof(float));
        assert( array[2*i + 1]->data );

        array[2*i + 1]->shape[0]     = n_out;
        array[2*i + 1]->ndim         = 1;
        array[2*i + 1]->endianness   = '<' ;  /* FIXME */
        array[2*i + 1]->typechar     = 'f' ;
        array[2*i + 1]->elem_size    = sizeof(float) ;
        array[2*i + 1]->fortran_order= false ;
        memset( array[2*i+1]->data, 0, n_out * sizeof(float));
#if 0
        c_npy_matrix_dump( array[2*i] );
        c_npy_matrix_dump( array[2*i+1] );
#endif
    }
    array[2*n_layers] = NULL;
    
    // printf("Length: %lu\n", c_npy_matrix_array_length( array ));
    int retval = c_npy_matrix_array_write( filename, array );
    if( retval != n_layers*2 )
        printf("Warning: Arrays written: %d  !=  2 x n_layers     (n_layers=%d)\n", retval, n_layers );

    for (int i = 0; i < 2*n_layers ; i++ ){
        free(array[i]->data);
        free(array[i]);
    }

    return retval;
}

int main(int argc, char *argv[] )
{
    if( argc < 5 ){
        printf("Usage: %s <savefilename.npz> n1 n2 <xavier|kaiming>\n", argv[0]);
        printf("   Or: %s <savefilename.npz> n1 n2 n3 <xavier|kaiming> <xavier|kaiming>\n", argv[0]);
    }
    srand(time(NULL));
    int n_layers = (argc / 2) - 1;
    int sizes[n_layers+1];
    for ( int i = 0; i <= n_layers; i++ )
        sizes[i] = atoi( argv[i+2] ); /* FIXME --- check (use strtol() ) */

    return create_and_save_random_weights( n_layers, sizes, argv + (n_layers+3), argv[1] );
}

