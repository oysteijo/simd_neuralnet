/* logger.c - Øystein Schønning-Johansen 2013 - 2023 */
/* 
 vim: ts=4 sw=4 softtabstop=4 expandtab 
*/
#include "logger.h"
#include "metrics.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>

struct _logger_t 
{
    callback_t  cb;
    /* Other data */
    int         epoch_count;
    const char *filename;
    bool        no_stdout;
};

static int find_last_epoch_from_logfile( logger_t *logger );

#define EPOCH_STR "Epoch"
#define MAX_BUFFER 255
/* Define and set the defaults. OMG, this is ugly but it is general. */
CALLBACK_DEFINE(logger,
        logger_config *cfg = (logger_config*) config;
        newcb->epoch_count = cfg->epoch_count;
        newcb->filename    = cfg->filename;
        newcb->no_stdout   = cfg->no_stdout;
        newcb->epoch_count = find_last_epoch_from_logfile( newcb );
);

/* Here is some ugly code that looks for a previous logfile and finds the epoch counter */
static int find_last_epoch_from_logfile( logger_t *logger )
{
    if( logger->epoch_count == 0 &&
        logger->filename    != NULL &&
        access(logger->filename, F_OK ) == 0 )
    {
        FILE *fp = fopen( logger->filename, "r");
        if(!fp)
            return logger->epoch_count; /* Fail silently should be OK */
        char buf[MAX_BUFFER + 1];
        fseek( fp, -MAX_BUFFER, SEEK_END );
        size_t len=fread( buf, 1, MAX_BUFFER, fp );
        fclose(fp);  /* Hurry up and close it */
        buf[len] = '\0';
        if( buf[len-1] == '\n' )  /* Dirty trick if the last line is terminated by newline (which it usually is) */
            buf[len-1] = '\0';
        char *last_newline = strrchr( buf, '\n' );
        if ( !last_newline )
            return logger->epoch_count;
        char *last_line = last_newline + 1;
        if( buf[len-1] == '\0' ) /* Fixing up the dirty trick */
            buf[len-1] = '\n';   /* we are working on a buffer copy anyway so this is not a big deal */
        char *e_counter = strstr( last_line, EPOCH_STR) + strlen(EPOCH_STR);
        return atoi( e_counter ) + 1;
    }
    return logger->epoch_count;
}

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
    ptr += sprintf ( ptr, EPOCH_STR " %3d ", log->epoch_count++ );

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

