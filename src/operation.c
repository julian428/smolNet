#include <lib/operation.h>

void free_operation(Operation* op){
	if(op) free(op);
}

int is_operation_nesecary(OperationType type, Tensor* input1, Tensor* input2, Tensor* result){
	if(!result) return 0;
	if(!result->creator) return 1;
	if(result->creator->type != type) return 1;
	if(result->creator->input1 != input1) return 1;
	if(result->creator->input2 != input2) return 1;
	return 0;
}

Operation* create_operation(OperationType type, Tensor* input1, Tensor* input2){
	Operation* op = (Operation*)malloc(sizeof(Operation));
	if(!op) return NULL;

	op->type = type;
	op->input1 = input1;
	op->input2 = input2;
	op->back = get_backwardoperation(type);
	if(input1) input1->ref_count++;
	if(input2) input2->ref_count++;

	return op;
}

BackwardOperation* get_backwardoperation(OperationType type){
	switch(type){
		case T_ADD:
			return backward_add_tensor;
		case T_MUL:
			return backward_mul_tensor;
		case T_DOT:
			return backward_dot_tensor;
		case T_SOFTMAX:
			return backward_softmax_tensor;
		case T_RELU:
			return backward_relu_tensor;
		case T_MSE:
			return backward_mse_tensor;
		case T_CEL:
			return backward_cel_tensor;
		default:
			return NULL;
	}
}

int twin_tensors(Tensor* a, Tensor* b){
	if(!a || !b || !a->shape || !b->shape) return 0;
	if(a->dims != b->dims) return 0;

	for(int i = 0; i < a->dims; i++){
		if(a->shape[i] != b->shape[i]) return 0;
	}

	return 1;
} 

int similar_tensors(Tensor* a, Tensor* b){
	if(a->batch_size % b->batch_size != 0 && b->batch_size % a->batch_size != 0) return 0;

	return 1;
}
