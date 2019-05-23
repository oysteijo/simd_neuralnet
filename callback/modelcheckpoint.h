#ifndef __MODELCHECKPOINT_H__
#define __MODELCHECKPOINT_H__
#include "optimizer.h"

typedef struct _checkpointdata_t 
{
    const char *filename;
    const int   monitor_idx;
    const bool  greater_is_better;
    const bool  verbose;
} checkpointdata_t;

void modelcheckpoint( const optimizer_t * opt, const float *epoch_results, bool validation_set_given, void *data );
#endif /* __MODELCHECKPOINT_H__ */

