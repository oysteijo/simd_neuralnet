/* This is the implementation of the abstract optimizer type */
#include "optimizer.h"

/* ---- Adam (Kingma and Ba, 2014) The name derives from "adaptive moments". ---- */
/* This implementation adds the Decoupled Weights Decay Regularization
 * introdused by Loshchilov and Hutter. https://arxiv.org/abs/1711.05101  */

OPTIMIZER_DECLARE(adam);
#define ADAM_OPTIMIZER(v) ((adam_t*)(v))

typedef struct _adam_properties_t {
    float learning_rate;
    float beta_1;
    float beta_2;
    float weight_decay;
} adam_properties_t;
#define ADAM_PROPERTIES(...) \
    &((adam_settings_t)  \
            { .learning_rate = 0.001f, .beta_1 = 0.9f, .beta_2 = 0.999f, .weight_decay = 1e-4f, __VA_ARGS__ })

