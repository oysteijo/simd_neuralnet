#include "earlystopping.h"

#include <stdio.h>
#include <float.h>
#include <assert.h>

void earlystopping(const optimizer_t * opt, const float *epoch_results, bool validation_set_given, void *data )
{
    earlystoppingdata_t *esdata = (earlystoppingdata_t*) data;

    static float best_so_far = FLT_MAX; 
    static int epochs_since_improvement = 0;

    bool improvement = false;
    /* which index? */
    int n_metrics = optimizer_get_n_metrics( opt );
    int idx = esdata->monitor_idx;
    if ( esdata->monitor_idx < 0 )
        idx = validation_set_given ? n_metrics : 0;
    /* Fixme. More checks */

    float score = epoch_results[idx];
    if (esdata->greater_is_better) {
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
    if ( epochs_since_improvement > esdata->patience ){
        esdata->early_stopping_flag = true;
    }

    return;
}

