/* This is the implementation of the abstract optimizer type */
#include "optimizer.h"

/* ---- Stochastic Gradient Decsent (SGD) ---- */
OPTIMIZER_DECLARE(SGD);
#define SGD_OPTIMIZER(v) ((SGD_t*)(v))

typedef struct _sgd_properties_t {
    float learning_rate;
    float decay;
    float momentum;
    bool  nesterov;
} sgd_properties_t;
#define SGD_PROPERTIES(...) \
    &((sgd_properties_t)  \
            { .learning_rate = 0.01f, .decay=0.0f, .momentum=0.0f, .nesterov=false, __VA_ARGS__ })

