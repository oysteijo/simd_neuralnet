/* neuralnet_predict_batch.h - Øystein Schønning-Johansen 2023 */
/* 
 vim: ts=4 sw=4 softtabstop=4 expandtab 
*/
#include "neuralnet.h"
void neuralnet_predict_batch( const neuralnet_t *nn, const int n_samples, const float *inputs, float *output );
