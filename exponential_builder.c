/* Comment here */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
/* Meta-code that generates exponential.h
 * Improvement suggestion: The step of the table is always 0.1. that can be parameterized.
 */
int main(int argc, char *argv[] ){
    
    if( argc != 2 ){
        fprintf(stderr, "This should generate 'exponential.h'. An integer argument gives the tabel size.\n");
        return -1;
    }

    int n = atoi(argv[1]);
    double x = 0.0;
    
    printf("/* exponential.h - Auto-generated file -- DO NOT EDIT */\n");
    printf("/* Auto-generated from code in '%s' */\n\n", __FILE__ );
    printf("#ifndef __EXPONENTIAL_H__\n#define __EXPONENTIAL_H__\n\n");
    printf("#define EXP_MAX_INDEX %d\n", n-1 );
    printf("#define EXP_MAX_VALUE %.9g\n\n", (n-1)/128.0 );
            
    printf("static float e[%d] = {", n );
    for( int i = 0; i < n; i++ ){
        if( (i & 0x03) == 0 ) printf("\n  ");
        printf("%.17gf, ", exp( x ) / 128.0 );
        x += 1.0 / 128.0;
    }
    printf("\n};\n");
    printf("#endif /* __EXPONENTIAL_H__ */\n");

    /* check */
    char max_value_str[20];
    sprintf( max_value_str, "%.9g", (n-1)/128.0 );
    float max_val = strtof( max_value_str, NULL );
    const int ck = (int) max_val * 128.0f;
    if( ck > (n-1) ){
        fprintf(stderr, "Max values looks up value beyond table.\n" );
        return -1;
    }
    return 0;
}

