#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <stdio.h>
#include <string.h>
#include <time.h>

typedef struct OperationS Operation;

typedef struct TensorS {
	double* data;
	double* grad;
	int* shape;
	int dims;
	int hyper_volume;
	char visited;
	Operation* creator;
} Tensor;

#include "utils.h"
#include "operation.h"

void free_tensor(Tensor* t);

Tensor* tensor_zeros(int* shape, int dims);
Tensor* tensor_random(int* shape, int dims);

void backward_tensor(Tensor* t);

//helpers
void print_tensor(Tensor* t);

#endif
