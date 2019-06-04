#ifndef __LOGGER_H__
#define __LOGGER_H__
#include "optimizer.h"

typedef struct _logdata_t 
{
    int         epoch_count;
    const char *filename;
    bool        no_stdout;
} logdata_t;

void logger( const optimizer_t * opt, const float *epoch_results, bool validation_set_given, void *data );
#endif /* __LOGGER_H__ */

