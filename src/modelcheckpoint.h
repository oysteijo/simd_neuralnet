#ifndef __MODELCHECKPOINT_H__
#define __MODELCHECKPOINT_H__
#include "callback.h"

CALLBACK_DECLARE(modelcheckpoint);

typedef struct _modelcheckpoint_config 
{
    const char *filename;
    const int   monitor_idx;
    const bool  greater_is_better;
    const bool  verbose;
} modelcheckpoint_config;

/* The 'macros' */
#define MODELCHECKPOINT(x) ((modelcheckpoint_t*)x)
#define MODELCHECKPOINT_NEW(...) \
    &((modelcheckpoint_config) { .filename=NULL, .monitor_idx=-1, .greater_is_better=false, .verbose=false, __VA_ARGS__ })

#endif /* __MODELCHECKPOINT_H__ */

