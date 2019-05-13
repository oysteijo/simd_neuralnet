#ifndef __EVALUATE_H__
#define __EVALUATE_H__
#include "neuralnet.h"
#include "metrics.h"

void evaluate( neuralnet_t *nn, const int n_valid_samples, const float *valid_X, const float *valid_Y,
        metric_func metrics[], float *results );
#endif  /* __EVALUATE_H__ */
