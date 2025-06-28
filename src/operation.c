#include <lib/operation.h>

void free_operation(Operation* op){
	if(op) free(op);
}

Operation* create_operation(OperationType type, Tensor* input1, Tensor* input2){
	Operation* op = (Operation*)malloc(sizeof(Operation));
	if(!op) return NULL;

	op->type = type;
	op->input1 = input1;
	op->input2 = input2;
	op->back = get_backwardoperation(type);

	return op;
}

int similar_tensors(Tensor* a, Tensor* b){
	if(!a || !b) return 0;
	if(a->dims != b->dims) return 0;

	for(int i = 0; i < a->dims; i++){
		if(a->shape[i] != b->shape[i]) return 0;
	}

	return 1;
}

Tensor* add_tensors(Tensor* a, Tensor* b){
	if(!similar_tensors(a, b)){
		fprintf(stderr, "Tensors %p and %p have different dimensions.\n", a, b);
		exit(EXIT_FAILURE);
	}

	Tensor* c = tensor_zeros(a->shape, a->dims);
	if(!c) return NULL;

	c->creator = create_operation(T_ADD, a, b);

	for(int i = 0 ; i < c->hyper_volume; i++){
		c->data[i] = a->data[i] + b->data[i];
	}

	return c;
}

void backward_add_tensor(Tensor* self){
	if(!self || !self->creator) return;

	Tensor* a = self->creator->input1;
	Tensor* b = self->creator->input2;

	for(int i = 0; i < self->hyper_volume; i++){
		a->grad[i] += self->grad[i];
		b->grad[i] += self->grad[i];
	}
}

Tensor* mul_tensors(Tensor* a, Tensor* b){
	if(!similar_tensors(a, b)){
		fprintf(stderr, "Tensors %p and %p have different dimensions.\n", a, b);
		exit(EXIT_FAILURE);
	}

	Tensor*c = tensor_zeros(a->shape, a->dims);
	if(!c) return NULL;

	c->creator = create_operation(T_MUL, a, b);

	for(int i = 0; i < c->hyper_volume; i++){
		c->data[i] = a->data[i] * b->data[i];
	}

	return c;
}

void backward_mul_tensor(Tensor* self){
	if(!self || !self->creator) return;

	Tensor* a = self->creator->input1;
	Tensor* b = self->creator->input2;

	for(int i = 0; i < self->hyper_volume; i++){
		a->grad[i] = self->grad[i] * b->data[i];
		b->grad[i] = self->grad[i] * a->data[i];
	}
}

Tensor* dot_tensors(Tensor* a, Tensor* b){
	if(a->dims != b->dims || a->dims < 2){
		fprintf(stderr, "Tensor %p or %p have only one dimension or aren't equal.\n", a, b);
		exit(EXIT_FAILURE);
	}

	int a_width = a->shape[a->dims-1];
	int a_height = a->shape[a->dims-2];
	int b_width = b->shape[b->dims-1];

	if(a_width != b->shape[b->dims - 2]){
		fprintf(stderr, "Inner dimensions of %p and %p don't match for mat_mul.\n", a, b);
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < a->dims-2; i++){
		if(a->shape[i] != b->shape[i]){
			fprintf(stderr, "Batch dimensions of %p and %p don't match for dot product.\n", a, b);
			exit(EXIT_FAILURE);
		}
	}

	int* new_shape = (int*)malloc(a->dims * sizeof(int));
	for(int i = 0; i < a->dims-2; i++) new_shape[i] = a->shape[i];
	new_shape[a->dims-2] = a_height; 
	new_shape[a->dims-1] = b_width;

	Tensor* c = tensor_zeros(new_shape, a->dims);
	free(new_shape);

	c->creator = create_operation(T_DOT, a, b);
	
	int batch_size = c->hyper_volume / (a_height * b_width);
	for(int i = 0; i < batch_size; i++){
		mat_mul(
			&a->data[i * a_width * a_height],
			&b->data[i * a_width * b_width],
			&c->data[i * a_height * b_width],
			a_width, a_height, b_width,
			index_regular, index_regular
		);
	}

	return c;
}

void backward_dot_tensor(Tensor* self){
	if(!self || !self->creator) return;

	Tensor* a = self->creator->input1;
	Tensor* b = self->creator->input2;

	int dims = self->dims;
	int a_height = a->shape[dims-2];
	int a_width = a->shape[dims-1];
	int b_width = b->shape[dims-1];
	int batch_size = self->hyper_volume / (a_height * b_width);

	for(int i = 0; i < batch_size; i++){
		double* dA = &a->grad[i * a_height * a_width];
		double* dB = &b->grad[i * a_width * b_width];
		double* dC = &self->grad[i * a_height * b_width];
		double* A = &a->data[i * a_height * a_width];
		double* B = &b->data[i * a_width * b_width];

		mat_mul(dC, B, dA, b_width, a_height, a_width, index_regular, index_transposed);
		mat_mul(A, dC, dB, a_height, a_width, b_width, index_transposed, index_regular);

	}
}

BackwardOperation* get_backwardoperation(OperationType type){
	switch(type){
		case T_ADD:
			return backward_add_tensor;
		case T_MUL:
			return backward_mul_tensor;
		case T_DOT:
			return backward_dot_tensor;
		default:
			return NULL;
	}
}
