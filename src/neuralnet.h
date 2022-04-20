#ifndef __NN_NEURALNET_H__
#define __NN_NEURALNET_H__

#define NN_MAX_LAYERS 8

typedef struct _neuralnet_t neuralnet_t;
typedef struct _layer_t layer_t;

struct _layer_t
{
    int    n_input, n_output;
    float *weight, *bias;
    void    (*activation_func) (const int n, float *ar);
#ifndef PREDICTION_ONLY
    void    (*activation_derivative) (const int n, const float *activation, float *ar);
#endif
    int    n_padding;
};

struct _neuralnet_t
{
    int     n_layers;
    layer_t layer[NN_MAX_LAYERS];
#ifndef PREDICTION_ONLY
    void    (*loss)  (const unsigned int n, const float *y_pred, const float *y_true, float *loss );
#endif
};

neuralnet_t * neuralnet_load             ( const char *filename );
void          neuralnet_free             (       neuralnet_t *nn); 
void          neuralnet_predict          ( const neuralnet_t *nn, const float *input, float *output);
#ifndef PREDICTION_ONLY
/* Two macros to hide the compound literals */
#define INT_ARRAY(...) (int[]){__VA_ARGS__}
#define STR_ARRAY(...) (char*[]){__VA_ARGS__, NULL}
neuralnet_t * neuralnet_create           ( const int n_layers, int sizes[], char *activation_funcs[] );
void          neuralnet_initialize       (       neuralnet_t *nn, char *initializers[] );
void          neuralnet_set_loss         (       neuralnet_t *nn, const char *loss_name );
void          neuralnet_backpropagation  ( const neuralnet_t *nn, const float *input, const float *desired, float *gradient);
void          neuralnet_save             ( const neuralnet_t *nn, const char *fmt, ...);
void          neuralnet_update           (       neuralnet_t *nn, const float *delta_w );
void          neuralnet_get_parameters   ( const neuralnet_t *nn, float *params );
#endif

static inline int neuralnet_get_n_layers ( const neuralnet_t *nn ) { return nn->n_layers; }

static inline
unsigned int neuralnet_total_n_parameters( const neuralnet_t *nn )
{
    unsigned int count = 0;
    for ( int i = 0; i < nn->n_layers; i++ )
        count += (nn->layer[i].n_input + 1) * nn->layer[i].n_output;
    return count;
}
#endif /* __NN_NEURALNET_H__ */
