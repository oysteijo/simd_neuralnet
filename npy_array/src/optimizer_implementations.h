#ifndef __OPTIMIZER_IMPLEMENTATIONS_H__
#define __OPTIMIZER_IMPLEMENTATIONS_H__
#include "optimizer.h"

#define EPOCH_RUNNER_DECLARE(opt_name) \
void opt_name ## _run_epoch( optimizer_t *opt, \
        const unsigned int n_train_samples, const float *train_X, const float *train_Y )

/* ---- Stochastic Gradient Decsent (SGD) ---- */
typedef struct _sgd_settings_t sgd_settings_t; 
struct _sgd_settings_t
{
    float learning_rate;
    float decay;
    float momentum;
    bool  nesterov;
};

EPOCH_RUNNER_DECLARE(SGD);
#define SGD_SETTINGS(...) \
    &((sgd_settings_t)  \
            { .learning_rate = 0.01f, .decay=0.0f, .momentum=0.0f, .nesterov=false, __VA_ARGS__ })

/* ---- RMSprop ---- */
typedef struct _RMSprop_settings_t RMSprop_settings_t; 
struct _RMSprop_settings_t
{
    float learning_rate;
    float rho;
    float decay;
    float momentum;
    bool  nesterov;
};

EPOCH_RUNNER_DECLARE(RMSprop);
#define RMSPROP_SETTINGS(...) \
    &((RMSprop_settings_t)  \
            { .learning_rate = 0.001f, \
              .rho           = 0.9f , \
              .decay         = 0.0f , \
              .momentum      = 0.0f , \
              .nesterov      = false, \
              __VA_ARGS__ })

/* ---- Adagrad ---- */
typedef struct _adagrad_settings_t adagrad_settings_t; 
struct _adagrad_settings_t
{
    float learning_rate;
    float decay;
};

EPOCH_RUNNER_DECLARE(adagrad);
#define ADAGRAD_SETTINGS(...) \
    &((adagrad_settings_t)  \
            { .learning_rate = 0.01f, .decay = 0.0f, __VA_ARGS__ })

/* ---- Adam ---- */
typedef struct _adam_settings_t adam_settings_t; 
struct _adam_settings_t
{
    float learning_rate;  /* Called step size in Goodfellow et al. */
    const float beta_1, beta_2; /* Called rho_1 and rho2 in Goodfellow et al., but this takes name from keras code. */
};
EPOCH_RUNNER_DECLARE(adam);
#define ADAM_SETTINGS(...) \
    &((adam_settings_t)  \
            { .learning_rate = 0.001f, .beta_1 = 0.9f, .beta_2 = 0.999f, __VA_ARGS__ })

/* ---- AdamW ---- */
typedef struct _adamw_settings_t adamw_settings_t; 
struct _adamw_settings_t
{
    float learning_rate;  /* Called step size in Goodfellow et al. */
    const float beta_1, beta_2; /* Called rho_1 and rho2 in Goodfellow et al., but this takes name from keras code. */
    float weight_decay;
};
EPOCH_RUNNER_DECLARE(adamw);
#define ADAMW_SETTINGS(...) \
    &((adamw_settings_t)  \
            { .learning_rate = 0.001f, .beta_1 = 0.9f, .beta_2 = 0.999f, .weight_decay = 1e-4f, __VA_ARGS__ })

#endif /* __OPTIMIZER_IMPLEMENTATIONS_H__ */
