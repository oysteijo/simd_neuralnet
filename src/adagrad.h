/* This is the implementation of the abstract optimizer type */
#include "optimizer.h"

/* ---- AdaGrad (Duchi et al. 2014) ---- */
OPTIMIZER_DECLARE(adagrad);
#define ADAGRAD_OPTIMIZER(v) ((adagrad_t*)(v))

typedef struct _adagrad_properties_t {
    float learning_rate;
    float decay;
} adagrad_properties_t;
#define ADAGRAD_PROPERTIES(...) \
    &((adagrad_properties_t)  \
            { .learning_rate = 0.01f, .decay = 0.0f, __VA_ARGS__ })

