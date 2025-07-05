#include <lib/layer.h>

void free_layer(Layer* l){
	if(!l) return;
	assert(l->output->owner == l);
	if(l->output) free_tensor(l->output);
	if(l->context) free(l->context);
	free(l);
}

Tensor* create_layer_tensor(Layer* l, int* shape, int dims){
	if(!l->output){
		Tensor* output = tensor_zeros(shape, dims);
		if(output) output->owner = l;
		return output;
	}
	if(!equal_vectors(l->output->shape, l->output->dims, shape, dims)){
		free_tensor(l->output);
		Tensor* output = tensor_zeros(shape, dims);
		if(output) output->owner = l;
		return output;
	}
	return l->output;
}
