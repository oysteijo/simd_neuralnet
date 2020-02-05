#include "neuralnet.h"
#include <stdio.h>
// #include "test.h"
//
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
    fprintf(stderr, "%-70s: %s\n", msg , a==b ? OK : FAIL); \
    if( a != b ) fail_count++;

#define CHECK_NOT_NULL_MSG(a,msg)               \
    test_count++;                                   \
    fprintf(stderr, "%-70s: %s\n", msg , a!=NULL ? OK : FAIL); \
    if( a == NULL ) fail_count++;

#define CHECK_STR_EQUALS_MSG(a,b,msg)               \
    test_count++;                                   \
    fprintf(stderr, "%-70s: %s\n", msg , strcmp(a,b)==0 ? OK : FAIL); \
    if( strcmp(a,b)==0 ) fail_count++;


int main(int argc, char *argv[] )
{
    int test_count = 0;
    int fail_count = 0;

    neuralnet_t *nn = neuralnet_create( 2,
            INT_ARRAY( 3,4,2 ),
            STR_ARRAY( "tanh", "sigmoid" ));

    CHECK_NOT_NULL_MSG( nn,
            "Checking that neural network was created" );
    
    CHECK_INT_EQUALS_MSG( 26, neuralnet_total_n_parameters(nn),
            "Checking that total number of parametes are 26" );
            
    CHECK_INT_EQUALS_MSG( 2, neuralnet_get_n_layers(nn),
            "Checking that total number of layers are 2" );
            
    neuralnet_free( nn );

    nn = neuralnet_create( 2,
            INT_ARRAY( 3,4,2, -1 ),
            STR_ARRAY( NULL, "sigmoid", "blah" )); /* Should work? */

    CHECK_NOT_NULL_MSG( nn,
            "Checking that neural network was created" );
    
    CHECK_INT_EQUALS_MSG( 26, neuralnet_total_n_parameters(nn),
            "Checking that total number of parametes are 26" );
            
    CHECK_INT_EQUALS_MSG( 2, neuralnet_get_n_layers(nn),
            "Checking that total number of layers are 2" );
            
    neuralnet_free( nn );

    printf("Total test done  : %d\n", test_count );
    printf("Total test failed: %d\n", fail_count );
    return 0;
}
    
