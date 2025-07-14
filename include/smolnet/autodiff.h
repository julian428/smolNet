#ifndef _AUTODIFF_H_
#define _AUTODIFF_H_

typedef struct TensorGraph {
	Tensor_sn** tensors;
	void (*free)(struct TensorGraph*);
	
	int graph_capacity;
	int tensor_count;
} TensorGraph_sn;

#endif
