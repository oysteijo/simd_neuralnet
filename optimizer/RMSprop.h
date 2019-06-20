#ifndef __RMSPROP_H__
#define __RMSPROP_H__
#include "optimizer.h"

typedef struct _RMSprop_settings_t RMSprop_settings_t; 
struct _RMSprop_settings_t
{
    float learning_rate;
    float rho;
    float decay;
    float momentum;
    bool  nesterov;
};

void RMSprop_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y );

/* Discuss: It is not really "setting", as the values in these struct of variables may change over the epochs.
 * Maybe a better name would be *_PARAMS(...) ?  */
#define RMSPROP_SETTINGS(...) \
    &((RMSprop_settings_t)  \
            { .learning_rate = 0.01f, \
              .rho           = 0.9f , \
              .decay         = 0.0f , \
              .momentum      = 0.0f , \
              .nesterov      = false, \
              __VA_ARGS__ })

#endif /* __RMSPROP_H__ */

