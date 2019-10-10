#include "logger.h"
#include "metrics.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>

struct _logger_t 
{
    callback_t  cb;
    /* Other data */
    int         epoch_count;
    const char *filename;
    bool        no_stdout;
};

/* Define and set the defaults. OMG, this is ugly but it is general. */
CALLBACK_DEFINE(logger,
        logger_config *cfg = (logger_config*) config;
        newcb->epoch_count = cfg->epoch_count;
        newcb->filename    = cfg->filename;
        newcb->no_stdout   = cfg->no_stdout;
);

#define MAX_BUFFER 255
void logger_callback_run( callback_t *cb, optimizer_t * opt, const float *epoch_results, bool validation_set_given )
{
    logger_t *log = (logger_t*) cb;

    char buffer[MAX_BUFFER + 1];
    char *ptr = buffer;
    time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );

    /* First the time, let's settle for the time of day, there will be other evidence to se the date and year.*/
    ptr += sprintf ( ptr, "[%02d:%02d:%02d] ",timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

    /* Then the epoch count. The count is only stored in log. It is not in the optimizer */
    ptr += sprintf ( ptr, "Epoch %3d ", log->epoch_count++ );

    /* Now the metric values */
    int n_metrics = optimizer_get_n_metrics( opt ); // * ( 1 + (int)validation_set_given);
    for( int phase = 0; phase <= ((int) validation_set_given); phase++ ){  /* phase 0 -> train error / phase 1: validation error */
        for ( int i = 0; i < n_metrics; i++ ){
            ptr += sprintf ( ptr, "%s%s: %5.5e ", phase == 1 ? "val_" :"", get_metric_name(opt->metrics[i]),
                    epoch_results[i + (phase * n_metrics)] );
        }
    }

    ptr += sprintf(ptr, "\n" );
    assert( ptr - buffer < MAX_BUFFER );

    /* Print it out */
    if ( !log->no_stdout )
        fprintf( stdout, "%s", buffer ); 

    FILE *fp = NULL;
    if ( log->filename )
        fp = fopen( log->filename, "a" );

    if ( fp ){
        fprintf( fp, "%s", buffer );
        fclose( fp );
    }
    return;
}

