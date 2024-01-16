/* This is the implementation of the abstract optimizer type */
#include "optimizer.h"

OPTIMIZER_DECLARE(SGD);
#define SGD_OPTIMIZER(v) ((SGD_t*)(v))

/* ---- Stochastic Gradient Decsent (SGD) ---- */
typedef struct _sgd_settings_t {
    float learning_rate;
    float decay;
    float momentum;
    bool  nesterov;
} sgd_settings_t;
#define SGD_SETTINGS(...) \
    &((sgd_settings_t)  \
            { .learning_rate = 0.01f, .decay=0.0f, .momentum=0.0f, .nesterov=false, __VA_ARGS__ })

