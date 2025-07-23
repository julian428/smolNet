#include <smolnet/layer.h>

Layer_sn* createDenseLayer(int input_size, int output_size){
	if(input_size < 1 || output_size < 1){
		assert(0);
		return NULL;
	}

	Layer_sn* layer = (Layer_sn*)malloc(sizeof(Layer_sn));
	if(!layer){
		assert(0);
		return NULL;
	}

	layer->free = freeDenseLayer;
	layer->erase = eraseDenseLayer;

	layer->type = L_DENSE;
	layer->param_count = 3;

	layer->getParameterRef = getDenseParameterRef;
	layer->forward = forwardDense;

	layer->output = NULL;

	layer->context = (void*)createDenseContext(input_size, output_size);
	if(!layer->context){
		freeDenseLayer(layer);
		assert(0);
		return NULL;
	}

	return layer;
}

void forwardDense(Layer_sn* layer, Tensor_sn* input){
	if(!layer || layer->type != L_DENSE) return;

	Tensor_sn* weights = *layer->getParameterRef(layer, 0);
	Tensor_sn* bias = *layer->getParameterRef(layer, 1);
	Tensor_sn** wxRef = layer->getParameterRef(layer, 2);
	
	if(*wxRef) releaseTensor(*wxRef);
	if(layer->output) releaseTensor(layer->output);

	*wxRef = tensorsOperation(weights, input, mul);
	layer->output = tensorsOperation(*wxRef, bias, add);
}

void freeDenseLayer(Layer_sn* layer){
	if(!layer || layer->type != L_DENSE) return;

	eraseDenseLayer(layer);
	free(layer);
}

void eraseDenseLayer(Layer_sn* layer){
	if(!layer || layer->type != L_DENSE) return;
	
	freeDenseContext(layer->context);
	eraseLayer(layer);
}

Tensor_sn** getDenseParameterRef(Layer_sn* layer, int index){
	if(!layer || layer->type != L_DENSE) return NULL;

	int prepared_index = index % layer->param_count;
	DenseContext_sn* ctx = (DenseContext_sn*)layer->context;
	if(!ctx) return NULL;

	switch(prepared_index){
		case 0:
			return &ctx->weights;
		case 1:
			return &ctx->bias;
		case 2:
			return &ctx->wx;
	}

	return NULL;
}

DenseContext_sn* createDenseContext(int input_size, int output_size){
	if(input_size < 1 || output_size < 1){
		assert(0);
		return NULL;
	}

	DenseContext_sn* ctx = (DenseContext_sn*)malloc(sizeof(DenseContext_sn));
	if(!ctx){
		assert(0);
		return NULL;
	}
	
	int weights_dims = 2;
	int* weights_shape = borrowInt(weights_dims);
	weights_shape[0] = input_size;
	weights_shape[1] = output_size;

	ctx->weights = borrowTensor(weights_dims, weights_shape);
	if(!ctx->weights){
		freeDenseContext(ctx);
		assert(0);
		return NULL;
	}

	int bias_dims = 1;
	int* bias_shape = borrowInt(bias_dims);
	bias_shape[0] = output_size;

	ctx->bias = borrowTensor(bias_dims, bias_shape);
	if(!ctx->bias){
		freeDenseContext(ctx);
		assert(0);
		return NULL;
	}

	ctx->wx = NULL;

	return ctx;
}

void freeDenseContext(DenseContext_sn* ctx){
	if(!ctx) return;

	releaseTensor(ctx->weights);
	releaseTensor(ctx->bias);
	releaseTensor(ctx->wx);

	free(ctx);
}
