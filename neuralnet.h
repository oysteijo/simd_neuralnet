#ifndef __MLP_NEURALNET_H__
#define __MLP_NEURALNET_H__

#define MLP_MAX_LAYERS 7

typedef struct _neuralnet_t neuralnet_t;
typedef struct _layer_t layer_t;

struct _layer_t
{
    int    n_input, n_output;
    float *weight, *bias;
    void    (*activation_func) (unsigned int n, float *ar);
};

struct _neuralnet_t
{
    int     n_layers;
    layer_t layer[MLP_MAX_LAYERS];
#if defined(TRAINING_FEATURES)
    void    (*train) (neuralnet_t *nn, const float *input, const float *desired, const void *data );
#endif
};

neuralnet_t * neuralnet_new              ( const char *filename);
void          neuralnet_free             (       neuralnet_t *nn); 
void          neuralnet_evaluate         ( const neuralnet_t *nn, const float *input, float *output);
#if defined(TRAINING_FEATURES)
typedef void (*trainfunc)                (       neuralnet_t *nn, const float *input, const float *desired, const void *data);
void          neuralnet_set_trainer      (       neuralnet_t *nn, trainfunc tf);
void          neuralnet_train            (       neuralnet_t *nn, const float *input, const float *desired, const void *data);
void          neuralnet_save             ( const neuralnet_t *nn, const char *filename);
void          neuralnet_scale_parameters ( const neuralnet_t *nn, int idx, float scale );
#endif

static inline int neuralnet_get_n_layers ( const neuralnet_t *mlp ) { return mlp->n_layers; }
/*
#define MLP_NEURALNET_GET_(type,val) \
static inline type neuralnet_get_ ## val ( const neuralnet_t *nn, int idx ) { return nn->val[idx]; }

MLP_NEURALNET_GET_(int    , layer_size);
MLP_NEURALNET_GET_(float *, weight); 
MLP_NEURALNET_GET_(float *, bias  ); 

static inline int neuralnet_get_layer_size_input  ( const neuralnet_t *mlp ){ return mlp->layer_size[0]; }
static inline int neuralnet_get_layer_size_output ( const neuralnet_t *mlp ){ return mlp->layer_size[mlp->n_layers - 1]; }

#undef MLP_NEURALNET_GET_
*/
#endif /* __MLP_NEURALNET_H__ */
