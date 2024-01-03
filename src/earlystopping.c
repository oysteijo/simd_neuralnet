/* earlystopping.c - Øystein Schønning-Johansen 2013 - 2023 */
/* 
 vim: ts=4 sw=4 softtabstop=4 expandtab 
*/
#include "earlystopping.h"
#include <stdio.h>
#include <float.h>

struct _earlystopping_t 
{
    callback_t  cb;
    /* Other data */
    int   patience;
    int   monitor_idx;
    bool  greater_is_better;
    bool  early_stopping_flag;
};

/* Define and set the defaults. OMG, this is ugly but it is general. */
CALLBACK_DEFINE(earlystopping,
        earlystopping_config *cfg  = (earlystopping_config*) config;
        newcb->patience            = cfg->patience;
        newcb->monitor_idx         = cfg->monitor_idx;
        newcb->greater_is_better   = cfg->greater_is_better;
        newcb->early_stopping_flag = false;
);

void earlystopping_callback_run( callback_t *cb, optimizer_t * opt, const float *epoch_results, bool validation_set_given )
{
    earlystopping_t *es = (earlystopping_t*) cb;

    static float best_so_far = FLT_MAX; 
    static int epochs_since_improvement = 0;

    bool improvement = false;
    /* which index? */
    int n_metrics = optimizer_get_n_metrics( opt );
    int idx = es->monitor_idx;
    if ( es->monitor_idx < 0 )
        idx = validation_set_given ? n_metrics : 0;
    /* Fixme. More checks */

    float score = epoch_results[idx];
    if (es->greater_is_better) {
        if ( score > best_so_far || best_so_far == FLT_MAX ){
            improvement = true;
            best_so_far = score;
        }
    } else {
        if( score < best_so_far ){
            improvement = true;
            best_so_far = score;
        }
    }

    if( improvement ){
        epochs_since_improvement = 0;
        return;
    }

    epochs_since_improvement++;
    if ( epochs_since_improvement > es->patience ){
        es->early_stopping_flag = true;
    }

    return;
}

bool earlystopping_do_stop( const earlystopping_t *es )
{
    return es ? es->early_stopping_flag : false;
}
