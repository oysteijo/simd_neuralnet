/* earlystopping.h - Øystein Schønning-Johansen 2020 - 2023 */
/*
  vim: ts=4 sw=4 softtabstop=4 expandtab 
 */
#ifndef __EARLYSTOPPING_H__
#define __EARLYSTOPPING_H__
#include "callback.h"

CALLBACK_DECLARE(earlystopping);

/* User configurables */
typedef struct _earlystopping_config
{
    const int  patience;
    const int  monitor_idx;
    const bool greater_is_better;
} earlystopping_config;

/* The 'macros' */
#define EARLYSTOPPING(x) ((earlystopping_t*)x)
#define EARLYSTOPPING_SETTINGS(...) \
    &((earlystopping_config) { .patience=10, .monitor_idx=-1, .greater_is_better=false, __VA_ARGS__ })

/* The 'methods' */
bool earlystopping_do_stop( const earlystopping_t *es );
#endif /* __EARLYSTOPPING_H__ */

