#include "optimizer.h"

typedef struct _sgd_settings_t sgd_settings_t; 
struct _sgd_settings_t
{
    float learning_rate;
    float decay;
    float momentum;
    bool  nesterov;
};

void SGD_run_epoch( optimizer_t *opt,
        const unsigned int n_train_samples, const float *train_X, const float *train_Y );

/* Discuss: It is not really "setting", as the values in these struct of variables may change over the epochs.
 * Maybe a better name would be SGD_PARAMS(...)  */
#define SGD_SETTINGS(...) \
    &((sgd_settings_t)  \
            { .learning_rate = 0.01f, .decay=0.0f, .momentum=0.0f, .nesterov=false, __VA_ARGS__ })

