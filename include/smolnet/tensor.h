#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

typedef struct Tensor Tensor_sn;
typedef struct Creator Creator_sn;
typedef void (tensorReleaseFunction)(Tensor_sn*);

#include "alloc.h"
#include "creator.h"

typedef struct Tensor {
	tensorReleaseFunction* free;
	tensorReleaseFunction* erase; // releases the data, grad and shape pointers
	Creator_sn* creator;

	float* data;
	float* grad;
	int* shape;

	int dims;
	int volume;
	int batches;
	int batch_size;

	int visited : 1;
} Tensor_sn;

Tensor_sn* createTensor();
Tensor_sn* createShapedTensor(int dims, ...);
tensorReleaseFunction freeTensor;
tensorReleaseFunction eraseTensor;

void printTensor(Tensor_sn* tensor);

#endif
