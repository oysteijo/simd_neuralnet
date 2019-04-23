#ifndef __NN_NEURALNET_H__
#define __NN_NEURALNET_H__

#define NN_MAX_LAYERS 7

typedef struct _neuralnet_t neuralnet_t;
typedef struct _layer_t layer_t;

struct _layer_t
{
    int    n_input, n_output;
    float *weight, *bias;
    void    (*activation_func) (unsigned int n, float *ar);
#if defined(TRAINING_FEATURES)
    void    (*activation_derivative) (unsigned int n, const float *activation, float *ar);
#endif
};

struct _neuralnet_t
{
    int     n_layers;
    layer_t layer[NN_MAX_LAYERS];
#if defined(TRAINING_FEATURES)
    void    (*loss)  (const unsigned int n, const float *y_pred, const float *y_true, float *loss );
#endif
};

neuralnet_t * neuralnet_new              ( const char *filename);
void          neuralnet_free             (       neuralnet_t *nn); 
void          neuralnet_predict          ( const neuralnet_t *nn, const float *input, float *output);
#if defined(TRAINING_FEATURES)
void          neuralnet_set_loss         (       neuralnet_t *nn, const char *loss_name );
void          neuralnet_backpropagation  ( const neuralnet_t *nn, const float *input, const float *desired, float *gradient);
void          neuralnet_save             ( const neuralnet_t *nn, const char *filename);
#endif

static inline int neuralnet_get_n_layers ( const neuralnet_t *nn ) { return nn->n_layers; }
#endif /* __NN_NEURALNET_H__ */
