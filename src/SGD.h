/*
 * =====================================================================================
 *
 *       Filename:  SGD.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/13/24 09:39:26
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "optimizer.h"
OPTIMIZER_DECLARE(SGD);

/* ---- Stochastic Gradient Decsent (SGD) ---- */

#define SGD_SETTINGS(...) \
    &((sgd_settings_t)  \
            { .learning_rate = 0.01f, .decay=0.0f, .momentum=0.0f, .nesterov=false, __VA_ARGS__ })


/*  SGD.c  */
struct _SGD_t 
{
    optimizer_t opt;
    /* Other data */
    float learning_rate;
    float decay;
    float momentum;
    bool  nesterov;
};

OPTIMIZER_DEFINE(SGD);

