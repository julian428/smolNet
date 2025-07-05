#include <lib/layer.h>

Tensor* softmax_forward(Layer* l, Tensor* input){
	if(!l || l->type != L_SOFTMAX) return NULL;
	l->output = create_layer_tensor(l, input->shape, input->dims);
	softmax_tensor(input, l->output);
	return l->output;
}

Layer* softmax_layer(int input){
	if(input < 1) return NULL;
	
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if(!l) return NULL;

	l->type = L_SOFTMAX;
	l->forward = softmax_forward;
	l->free = free_layer;
	l->context = NULL;
	l->param_count = 0;
	l->get_param = NULL;
	l->output = NULL;

	return l;
}

Tensor* relu_forward(Layer* l, Tensor* input){
	if(!l || l->type != L_RELU) return NULL;
	l->output = create_layer_tensor(l, input->shape, input->dims);
	relu_tensor(input, l->output);
	return l->output;
}

Layer* relu_layer(int input){
	if(input < 1) return NULL;
	
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if(!l) return NULL;

	l->type = L_RELU;
	l->forward = relu_forward;
	l->free = free_layer;
	l->context = NULL;
	l->param_count = 0;
	l->get_param = NULL;
	l->output = NULL;

	return l;
}

