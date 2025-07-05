#include <lib/operation.h>

//softmax
void softmax_tensor(Tensor* a, Tensor* b){
	if(!a || !b) return;

	if(is_operation_nesecary(T_SOFTMAX, a, NULL, b)){
		if(b->creator) free_operation(b->creator);
		b->creator = create_operation(T_SOFTMAX, a, NULL);
	}

	int input_length = a->shape[a->dims - 1];
	int batch_size = a->hyper_volume / input_length;

	for(int i = 0; i < batch_size; i++){
		softmax(&a->data[i*input_length], &b->data[i*input_length], input_length);
	}
}

void backward_softmax_tensor(Tensor* self){
	if (!self || !self->creator || !self->creator->input1) return;

	Tensor* input = self->creator->input1;
	int input_length = input->shape[input->dims - 1];
	int batch_size = input->hyper_volume / input_length;

	for (int b = 0; b < batch_size; b++) {
		double* s = &self->data[b * input_length];
		double* grad_out = &self->grad[b * input_length];

		for (int i = 0; i < input_length; i++) {
			double grad = 0.0;
			for (int j = 0; j < input_length; j++) {
				double delta = (i == j) ? 1.0 : 0.0;
				grad += grad_out[j] * s[i] * (delta - s[j]);
			}
			input->grad[b * input_length + i] += grad;
		}
	}
}

//relu
void relu_tensor(Tensor* a, Tensor* b){
	if(!a || !b) return;

	if(is_operation_nesecary(T_RELU, a, NULL, b)){
		if(b->creator) free_operation(b->creator);
		b->creator = create_operation(T_RELU, a, NULL);
	}

	for(int i = 0; i < a->hyper_volume; i++){
		b->data[i] = a->data[i] > 0 ? a->data[i] : 0;
	}
}

void backward_relu_tensor(Tensor* self){
	if(!self || !self->creator || !self->grad) return;

	Tensor* input = self->creator->input1;
	if(!input || !input->grad || !input->data) return;

	for(int i = 0; i < input->hyper_volume; i++){
		input->grad[i] += self->grad[i] * (input->data[i] > 0 ? 1 : 0);
	}
}
