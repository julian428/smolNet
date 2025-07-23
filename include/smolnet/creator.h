#ifndef _CREATOR_H_
#define _CREATOR_H_

#include <stdlib.h>
#include <assert.h>

typedef struct Tensor Tensor_sn;

typedef void (operationFunction)(
		Tensor_sn* a, Tensor_sn* b, Tensor_sn* res, int i_a, int i_b, int i_res
);

typedef struct Creator {
	Tensor_sn* dad;
	Tensor_sn* mom;
	
	operationFunction* back;
} Creator_sn;

#include "alloc.h"

Creator_sn* createCreator(Tensor_sn* mom, Tensor_sn* dad, operationFunction* revFunc);
void printCreator(Creator_sn* creator);

// basic
int* broadcastShape(Tensor_sn* a, Tensor_sn* b, int* dims);
int getBroadcastIndex(int idx, int original_dim);
Tensor_sn* broadcastedTensor(Tensor_sn* a, Tensor_sn* b);

Tensor_sn* tensorsOperation(Tensor_sn* a, Tensor_sn* b, operationFunction* opFunc);

// operations
operationFunction add;
operationFunction mul;

// backward operations
operationFunction backAdd;
operationFunction backMul;

#endif
