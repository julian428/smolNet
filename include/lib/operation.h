#ifndef _OPERATIONS_H_
#define _OPERATIONS_H_

#include <stdlib.h>

#define EPSILON 1e-12

typedef struct TensorS Tensor;

typedef void (BackwardOperation)(Tensor*);

typedef enum {
	T_ADD,
	T_MUL,
	T_DOT,
	T_SOFTMAX,
	T_RELU,
	T_MSE,
	T_CEL
} OperationType;

typedef struct OperationS {
	OperationType type;
	Tensor* input1;
	Tensor* input2;
	BackwardOperation* back;
} Operation;

#include "tensor.h"

void free_operation(Operation* op);
int twin_tensors(Tensor* a, Tensor* b);
int similar_tensors(Tensor* a, Tensor* b);
int is_operation_nesecary(OperationType type, Tensor* input1, Tensor* input2, Tensor* result);
Operation* create_operation(OperationType type, Tensor* input1, Tensor* input2);

void add_tensors(Tensor* a, Tensor* b, Tensor* c);
void mul_tensors(Tensor* a, Tensor* b, Tensor* c);
void dot_tensors(Tensor* a, Tensor* b, Tensor* c);
void softmax_tensor(Tensor* a, Tensor* b);
void relu_tensor(Tensor* a, Tensor* b);
void mse_tensors(Tensor* output, Tensor* expected, Tensor* loss);
void cel_tensors(Tensor* output, Tensor* expected, Tensor* loss);

BackwardOperation backward_add_tensor;
BackwardOperation backward_mul_tensor;
BackwardOperation backward_dot_tensor;
BackwardOperation backward_softmax_tensor;
BackwardOperation backward_relu_tensor;
BackwardOperation backward_mse_tensor;
BackwardOperation backward_cel_tensor;

BackwardOperation* get_backwardoperation(OperationType type);

#endif
