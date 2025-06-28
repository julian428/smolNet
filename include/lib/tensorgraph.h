#ifndef _TENSORLIST_H_
#define _TENSORLIST_H_

#include <stdlib.h>

#include "tensor.h"

typedef struct {
	Tensor** items;
	int capacity;
	int size;
} TensorGraph;

void free_tensorgraph(TensorGraph* graph);
void free_tensors(TensorGraph* graph);
void search_tensors(TensorGraph* graph, Tensor* tensor);
int resize_tensorgraph(TensorGraph* graph);
int unvisit_tensors(TensorGraph* graph);
int add_to_tensorgraph(TensorGraph* graph, Tensor* tensor);
TensorGraph* create_tensorgraph(Tensor* root);

#endif
