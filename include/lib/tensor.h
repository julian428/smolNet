#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

typedef struct OperationS Operation;
typedef struct LayerS Layer;

typedef struct TensorS {
	double* data;
	double* grad;
	int* shape;
	int dims;

	int hyper_volume;
	int batches;
	int batch_size;

	int visited : 1;
	int ref_count;
	int id;
	Operation* creator;
	Layer* owner;
} Tensor;

#include "utils.h"
#include "operation.h"

void free_tensor(Tensor* t);
void zero_gradient(Tensor* t);

Tensor* tensor_zeros(int* shape, int dims);
Tensor* tensor_random(int* shape, int dims);
Tensor* tensor_data(double* data, int* shape, int dims);

//helpers
void print_tensor(Tensor* t);

#endif
