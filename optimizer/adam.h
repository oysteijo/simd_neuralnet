#ifndef __ADAM_H__
#define __ADAM_H__
#include "optimizer.h"

typedef struct _adam_settings_t adam_settings_t; 
struct _adam_settings_t
{
    float learning_rate;  /* Called step size in Goodfellow et al. */
    const float beta_1, beta_2; /* Called rho_1 and rho2 in Goodfellow et al., but this takes name from keras code. */
};

void adam_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y );

/* Discuss: It is not really "setting", as the values in these struct of variables may change over the epochs.
 * Maybe a better name would be *_PARAMS(...) ?  */
#define ADAM_SETTINGS(...) \
    &((adam_settings_t)  \
            { .learning_rate = 0.001f, .beta_1 = 0.9f, .beta_2 = 0.999f, __VA_ARGS__ })

#endif /* __ADAM_H__ */

