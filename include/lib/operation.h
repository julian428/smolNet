#ifndef _OPERATIONS_H_
#define _OPERATIONS_H_

#include <stdlib.h>

typedef struct TensorS Tensor;

typedef void (BackwardOperation)(Tensor*);

typedef enum {
	T_ADD,
	T_MUL,
	T_DOT
} OperationType;

typedef struct OperationS {
	OperationType type;
	Tensor* input1;
	Tensor* input2;
	BackwardOperation* back;
} Operation;

#include "tensor.h"

void free_operation(Operation* op);
Operation* create_operation(OperationType type, Tensor* input1, Tensor* input2);

Tensor* add_tensors(Tensor* a, Tensor* b);
Tensor* mul_tensors(Tensor* a, Tensor* b);
Tensor* dot_tensors(Tensor* a, Tensor* b);

BackwardOperation backward_add_tensors;
BackwardOperation backward_mul_tensors;
BackwardOperation backward_dot_tensors;

BackwardOperation* get_backwardoperation(OperationType type);

#endif
