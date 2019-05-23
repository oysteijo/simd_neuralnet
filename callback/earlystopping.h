#ifndef __EARLYSTOPPING_H__
#define __EARLYSTOPPING_H__
#include "optimizer.h"

typedef struct _earlystoppingdata_t 
{
    int         patience;
    const int   monitor_idx;
    const bool  greater_is_better;
    bool        early_stopping_flag;
} earlystoppingdata_t;

void earlystopping( const optimizer_t * opt, const float *epoch_results, bool validation_set_given, void *data );
#endif /* __EARLYSTOPPING_H__ */

