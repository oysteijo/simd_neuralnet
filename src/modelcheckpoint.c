#include "modelcheckpoint.h"
#include "neuralnet.h"
#include "metrics.h"

#include <stdio.h>
#include <float.h>
#include <assert.h>

struct _modelcheckpoint_t
{
    callback_t  cb;
    /* Other data */
    const char *filename;
    int monitor_idx;
    bool greater_is_better;
    bool verbose;
};

/* Define and set the defaults. OMG, this is ugly but it is general. */
CALLBACK_DEFINE(modelcheckpoint,
        modelcheckpoint_config *cfg  = (modelcheckpoint_config*) config;
        newcb->filename              = cfg->filename;
        newcb->monitor_idx           = cfg->monitor_idx;
        newcb->greater_is_better     = cfg->greater_is_better;
        newcb->verbose               = cfg->verbose;
);

void modelcheckpoint_callback_run( callback_t *cb, optimizer_t * opt, const float *epoch_results, bool validation_set_given )
{
    modelcheckpoint_t *mcp = (modelcheckpoint_t*) cb;

    static float best_so_far = FLT_MAX; /* This will fail for greater_is_better */

    bool do_save = false;
    /* which index? */
    int n_metrics = optimizer_get_n_metrics( opt );
    int idx = mcp->monitor_idx;
    if ( mcp->monitor_idx < 0 )
        idx = validation_set_given ? n_metrics : 0;
    /* Fixme. More checks */

    float score = epoch_results[idx];
    if (mcp->greater_is_better) {
        if ( score > best_so_far || best_so_far == FLT_MAX ){
            if( mcp->verbose )  /* BUG: This gives bullshit the first time */
                printf("Improved from %.4e to %.4e (Saving checkpoint)\n", best_so_far, score );
            do_save = true;
            best_so_far = score;
        }
    } else {
        if( score < best_so_far ){
            if( mcp->verbose )
                printf("Improved from %.4e to %.4e (Saving checkpoint)\n", best_so_far, score );
            do_save = true;
            best_so_far = score;
        }
    }

    if( do_save )
        neuralnet_save( opt->nn, mcp->filename ? mcp->filename : "checkpoint.npz");
    return;
}

