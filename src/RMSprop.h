/* This is the implementation of the abstract optimizer type */
#include "optimizer.h"

/* ---- RMSProp (Hinton 2012) ---- */
OPTIMIZER_DECLARE(RMSprop);
#define RMSPROP_OPTIMIZER(v) ((RMSprop_t*)(v))

typedef struct _rmsprop_properties_t {
    float learning_rate;
    float rho;
    float decay;
    float momentum;
    bool  nesterov;
} rmsprop_properties_t;
#define RMSPROP_PROPERTIES(...) \
    &((rmsprop_properties_t)  \
            { .learning_rate = 0.001f, \
              .rho           = 0.9f , \
              .decay         = 0.0f , \
              .momentum      = 0.0f , \
              .nesterov      = false, \
              __VA_ARGS__ })

