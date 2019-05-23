#ifndef __ADAGRAD_H__
#define __ADAGRAD_H__
#include "optimizer.h"

typedef struct _adagrad_settings_t adagrad_settings_t; 
struct _adagrad_settings_t
{
    float learning_rate;
};

void adagrad_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y );

/* Discuss: It is not really "setting", as the values in these struct of variables may change over the epochs.
 * Maybe a better name would be *_PARAMS(...) ?  */
#define ADAGRAD_SETTINGS(...) \
    &((adagrad_settings_t)  \
            { .learning_rate = 0.01, __VA_ARGS__ })

#endif /* __ADAGRAD_H__ */

