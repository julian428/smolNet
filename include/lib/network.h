#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <stdarg.h>

#include "tensor.h"
#include "layer.h"
#include "tensorgraph.h"
#include "operation.h"

#define INITIAL_NETWORK_CAPACITY 8

typedef struct NetworkS {
	Layer** layers;
	Tensor* output;
	Tensor* loss;
	int length;
	int capacity;

	TensorGraph* graph;
} Network;

Network* network(int initial_capacity, ...);
Tensor* network_calculate_loss(Network* n, void (*loss_function)(Tensor*, Tensor*, Tensor*), Tensor* expected);

int add_layer(Network* n, Layer* l);
int network_resize(Network* n);
int pop_layer(Network* n);

void network_free(Network* n);
void network_forward(Network* n, Tensor* input);
void network_backward(Network* n);
void network_zero_gradients(Network* n);
void network_step(Network* n, void (*loss_function)(Tensor*, Tensor*, Tensor*), Tensor* input, Tensor* expected);

#endif
