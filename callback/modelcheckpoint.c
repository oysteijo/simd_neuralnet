#include "modelcheckpoint.h"
#include "neuralnet.h"
#include "metrics.h"

#include <stdio.h>
#include <float.h>
#include <assert.h>

void modelcheckpoint(const optimizer_t * opt, const float *epoch_results, bool validation_set_given, void *data )
{
    checkpointdata_t *checkpointdata = (checkpointdata_t*) data;

    static float best_so_far = FLT_MAX; /* This will fail for greater_is_better */

    bool do_save = false;
    /* which index? */
    int n_metrics = optimizer_get_n_metrics( opt );
    int idx = checkpointdata->monitor_idx;
    if ( checkpointdata->monitor_idx < 0 )
        idx = validation_set_given ? n_metrics : 0;
    /* Fixme. More checks */

    float score = epoch_results[idx];
    if (checkpointdata->greater_is_better) {
        if ( score > best_so_far || best_so_far == FLT_MAX ){
            if( checkpointdata->verbose )  /* BUG: THis gives bullshit the first time */
                printf("Improved from %.4e to %.4e (Saving checkpoint)\n", best_so_far, score );
            do_save = true;
            best_so_far = score;
        }
    } else {
        if( score < best_so_far ){
            if( checkpointdata->verbose )
                printf("Improved from %.4e to %.4e (Saving checkpoint)\n", best_so_far, score );
            do_save = true;
            best_so_far = score;
        }
    }

    if( do_save )
        neuralnet_save( opt->nn, checkpointdata->filename ? checkpointdata->filename : "checkpoint.npz");
    return;
}

