#include <lib/layer.h>

void free_denselayer(Layer* l){
	if(!l) return;
	if(l->type != L_DENSE){
		fprintf(stderr, "Layer %p is not dense. Couldn't free.\n", l);
		return;
	}
	
	DenseLayer* ctx = (DenseLayer*)l->context;
	if(ctx){
		if(ctx->weights && ctx->weights->owner == l) free_tensor(ctx->weights);
		if(ctx->bias && ctx->bias->owner == l) free_tensor(ctx->bias);
		if(ctx->wx && ctx->wx->owner == l) free_tensor(ctx->wx);
	}

	free_layer(l);
}

Tensor* dense_forward(Layer* l, Tensor* input){
	if(l->type != L_DENSE) return NULL;
	DenseLayer* ctx = (DenseLayer*)l->context;

	dot_tensors(input, ctx->weights, ctx->wx);
	add_tensors(ctx->wx, ctx->bias, l->output);
	return l->output;
}

Tensor* get_param_dense(Layer* l, int i){
	if(l->type != L_DENSE) return NULL;
	DenseLayer* ctx = (DenseLayer*)l->context;
	if(i % 2 == 0) return ctx->weights;
	else return ctx->bias;
}

Layer* dense_layer(int input, int output){
	if(input < 1 || input < 1) return NULL;

	Layer* l = (Layer*)malloc(sizeof(Layer));
	if(!l) return NULL;

	l->output = tensor_zeros((int[]){1, output}, 2);
	if(!l->output){
		free_denselayer(l);
		return NULL;
	}
	l->output->owner = l;

	l->context = (void*)malloc(sizeof(DenseLayer));
	if(!l->context){
		free_denselayer(l);
		return NULL;
	}

	DenseLayer* ctx = (DenseLayer*)l->context;

	ctx->weights = tensor_random((int[]){input, output}, 2);
	if(!ctx->weights){
		free_denselayer(l);
		return NULL;
	}
	ctx->weights->owner = l;

	ctx->bias = tensor_random((int[]){1, output}, 2);
	if(!ctx->bias){
		free_denselayer(l);
		return NULL;
	}
	ctx->bias->owner = l;

	ctx->wx = tensor_zeros((int[]){1, output}, 2);
	if(!ctx->wx){
		free_denselayer(l);
		return NULL;
	}
	ctx->wx->owner = l;
	
	l->type = L_DENSE;
	l->forward = dense_forward;
	l->free = free_denselayer;
	l->param_count = 2;
	l->get_param = get_param_dense;

	return l;
}
