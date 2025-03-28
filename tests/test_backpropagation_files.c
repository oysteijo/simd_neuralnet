#include "neuralnet.h"
#include "npy_array.h"
#include "npy_array_list.h"
#include "simd.h"
#include "metrics.h"
#include "loss.h"
#include "test.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

int main( int argc, char *argv[]) 
{
    /* This testing program could actually read a nn from a file.
     * Say you read command line inputs like:
     *
     *   ./test_backpropagation_files <file_with_nn.npz> <file_with_input.npy> <file_with_target.npy> --output=outputfile.txt
     *
     * BTW: all testfiles should actually take an option to redirect output.
     */

	if( argc != 5 ){
		fprintf(stderr, "%s <file_with_nn.npz> <file_with_samples.npz> <file_with_gradient.npz> loss\n", argv[0]);
		return -1;
	}

	neuralnet_t *nn = neuralnet_load( argv[1] );

	npy_array_list_t *sample = npy_array_list_load( argv[2] );
	npy_array_list_t *gradient  = npy_array_list_load( argv[3] );

	char *loss = argv[4];
	
	neuralnet_set_loss( nn, loss);
	npy_array_t *test_X = sample->array;
	npy_array_t *test_Y = sample->next->array;

	int n_param = neuralnet_total_n_parameters( nn );
	float SIMD_ALIGN(grad[n_param]);

	neuralnet_backpropagation( nn, (float*) test_X->data, (float*) test_Y->data, grad);

	float *our_grad = grad;
	int n_tests = 0;
	int n_fail = 0;
    int print_col = 0;
	float epsilon = 0.0001f;
	float abs_error_sum = 0.0f;
	for ( npy_array_list_t *p = gradient; p; p = p->next->next ){
		float *w_grad = (float*) p->array->data;
		float *b_grad = (float*) p->next->array->data;
		/* Bias first */
		for ( int i = 0; i < (int)p->next->array->shape[0]; i++ ){
			float filegrad = *b_grad++;
			float ourgrad = *our_grad++;
			float abs_err = fabsf( filegrad - ourgrad );
			abs_error_sum += abs_err;
			if( fabsf( abs_err ) > epsilon ){
				fprintf( stderr, KRED "\ngrad0 != file value   (%5.5f != %5.5f)\n" KNRM, ourgrad, filegrad );
				n_fail++;
				print_col = 0;
			} else {
				fprintf( stderr, KGRN "." KNRM );
				print_col++;
			}
			if( ((print_col+1) % 64 == 0) ){
				fprintf( stderr, "\n" );
				print_col = 0;
			}
			n_tests++;
		}
		/* weights */
		for ( int i = 0; i < (int)(p->array->shape[0] * p->array->shape[1]); i++ ){
			float filegrad = *w_grad++;
			float ourgrad = *our_grad++;
			float abs_err = fabsf( filegrad - ourgrad );
			abs_error_sum += abs_err;
			if( fabsf( abs_err ) > epsilon ){
				fprintf( stderr, KRED "\ngrad0 != file value   (%5.5f != %5.5f)\n" KNRM, ourgrad, filegrad );
				n_fail++;
				print_col = 0;
			} else {
				fprintf( stderr, KGRN "." KNRM );
				print_col++;
			}
			if( ((print_col+1) % 64 == 0) ){
				fprintf( stderr, "\n" );
				print_col = 0;
			}
			n_tests++;
		}
	}
    fprintf(stderr, "\n" );

    fprintf(stderr, "Gradient test report, using data files and epsilon=%g\n", epsilon);
    fprintf(stderr, "%5d tests done\n", n_tests );
    fprintf(stderr, "%s%5d tests failed\n" KNRM, n_fail ? KRED : KGRN, n_fail );
	fprintf(stderr, "Mean abs. error of all gradient values: %g\n", abs_error_sum/n_tests ); 
	return 0;
}
