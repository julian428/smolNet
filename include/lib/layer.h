#ifndef _LAYER_H_
#define _LAYER_H_

#include <stdio.h>
#include <assert.h>
#include "utils.h"
#include "tensor.h"

typedef struct LayerS Layer;
typedef Tensor* (propagation_function)(Layer*, Tensor*);
typedef Tensor* (get_param)(Layer*, int);

typedef enum {
	L_DENSE,
	L_SOFTMAX,
	L_RELU
} LayerType;

typedef struct LayerS {
	LayerType type;
	void* context;
	Tensor* output;
	propagation_function* forward;
	int param_count;
	get_param* get_param;
	void (*free)(Layer*);
} Layer;

void free_layer(Layer* l);
Tensor* create_layer_tensor(Layer* l, int* shape, int dims);

// dense layer
typedef struct {
	Tensor* weights;
	Tensor* wx;
	Tensor* bias;
} DenseLayer;

void free_denselayer(Layer* l);
Tensor* get_param_dense(Layer* l, int i);
propagation_function dense_forward;
Layer* dense_layer(int input, int output);

// activation layers
propagation_function softmax_forward;
Layer* softmax_layer(int input);

propagation_function relu_forward;
Layer* relu_layer(int input);

#endif
