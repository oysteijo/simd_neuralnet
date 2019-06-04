#ifndef __ADAMW_H__
#define __ADAMW_H__
#include "optimizer.h"

typedef struct _adamw_settings_t adamw_settings_t; 
struct _adamw_settings_t
{
    float learning_rate;  /* Called step size in Goodfellow et al. */
    const float beta_1, beta_2; /* Called rho_1 and rho2 in Goodfellow et al., but this takes name from keras code. */
    float weight_decay;
};

void adamw_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y );

/* Discuss: It is not really "setting", as the values in these struct of variables may change over the epochs.
 * Maybe a better name would be *_PARAMS(...) ?  */
#define ADAMW_SETTINGS(...) \
    &((adamw_settings_t)  \
            { .learning_rate = 0.001, .beta_1 = 0.9f, .beta_2 = 0.999f, .weight_decay = 1e-4f, __VA_ARGS__ })

#endif /* __ADAMW_H__ */

