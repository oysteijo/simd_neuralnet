#include "optimizer.h"

OPTIMIZER_DECLARE(SGD);

struct _SGD_t {
   optimizer_t opt;
   /* other data */
   float learning_rate;
   float decay;
   /* float momentum; */
   /* bool nesterov; */
};

