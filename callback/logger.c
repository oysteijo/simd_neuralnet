#include "logger.h"
#include "metrics.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>

#define MAX_BUFFER 255
void logger(const optimizer_t * opt, const float *epoch_results, bool validation_set_given, void *data )
{
    logdata_t *logdata = (logdata_t*) data;
    char buffer[MAX_BUFFER + 1];
    char *ptr = buffer;
    time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );

    /* First the time, let's settle for the time of day, there will be other evidence to se the date and year.*/
    ptr += sprintf ( ptr, "[%02d:%02d:%02d] ",timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

    /* Then the epoch count. The count is only stored in logdata. It is not in the optimizer */
    ptr += sprintf ( ptr, "Epoch %3d ", logdata->epoch_count++ );

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
    if ( !logdata->no_stdout )
        fprintf( stdout, "%s", buffer ); 

    FILE *fp = NULL;
    if ( logdata->filename )
        fp = fopen( logdata->filename, "a" );

    if ( fp ){
        fprintf( fp, "%s", buffer );
        fclose( fp );
    }
    return;
}

