#ifndef _CREATOR_H_
#define _CREATOR_H_

#include <stdlib.h>
#include <assert.h>

typedef struct Tensor Tensor_sn;

typedef void (backwardFunction)(Tensor_sn*);
typedef enum {
	OP_NONE,
	OP_ADD
} CreatorType;

typedef struct Creator {
	Tensor_sn* dad;
	Tensor_sn* mom;
	
	backwardFunction* back;

	CreatorType type;
} Creator_sn;

#include "alloc.h"

Creator_sn* createCreator(Tensor_sn* mom, Tensor_sn* dad, CreatorType type);
void printCreator(Creator_sn* creator);

// basic
int broadcast_shape(Tensor_sn* a, Tensor_sn* b, int* output_shape);
int get_broadcast_index(int idx, int original_dim);

Tensor_sn* add_tensors(Tensor_sn* a, Tensor_sn* b);
Tensor_sn* mul_tensors(Tensor_sn* a, Tensor_sn* b);

#endif
