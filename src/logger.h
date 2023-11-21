/* logger.h - Øystein Schønning-Johansen 2020 - 2023 */
/*
  vim: ts=4 sw=4 softtabstop=4 expandtab 
 */
#ifndef __LOGGER_H__
#define __LOGGER_H__
#include "callback.h"

CALLBACK_DECLARE(logger);

typedef struct _logger_config
{
    const int   epoch_count;
    const char *filename;
    const bool  no_stdout;
} logger_config;

/* The 'macros' */
#define LOGGER(x) ((logger_t*)x)
#define LOGGER_NEW(...) \
    &((logger_config) { .epoch_count=0, .filename=NULL, .no_stdout=false, __VA_ARGS__ })

#endif /* __LOGGER_H__ */

