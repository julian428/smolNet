#include <smolnet/layer.h>

Layer_sn* createDenseLayer(int batch_count, int input_size, int output_size){
	if(batch_count < 1 || input_size < 1 || output_size < 1){
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

	layer->getParameter = getDenseParameter;
	layer->forward = forwardDense;

	int output_dims = 2;
	int* output_shape = borrowInt(output_dims);
	output_shape[0] = batch_count;
	output_shape[1] = output_size;

	layer->output = borrowTensor(output_dims, output_shape);
	if(!layer->output){
		freeDenseLayer(layer);
		assert(0);
		return NULL;
	}

	layer->context = (void*)createDenseContext(batch_count, input_size, output_size);
	if(!layer->context){
		freeDenseLayer(layer);
		assert(0);
		return NULL;
	}

	return layer;
}

void forwardDense(Layer_sn* layer, Tensor_sn* input){
	if(!layer || layer->type != L_DENSE) return;

	(void)layer; (void)input;
	assert(0 && "Not yet implemented.");
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

Tensor_sn* getDenseParameter(Layer_sn* layer, int index){
	if(!layer || layer->type != L_DENSE) return NULL;

	int prepared_index = index % layer->param_count;
	DenseContext_sn* ctx = (DenseContext_sn*)layer->context;
	if(!ctx) return NULL;

	switch(prepared_index){
		case 0:
			return ctx->weights;
		case 1:
			return ctx->bias;
		case 2:
			return ctx->wx;
	}

	return NULL;
}

DenseContext_sn* createDenseContext(int batch_count, int input_size, int output_size){
	if(batch_count < 1 || input_size < 1 || output_size < 1){
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

	int wx_dims = 2;
	int* wx_shape = borrowInt(wx_dims);
	wx_shape[0] = batch_count;
	wx_shape[1] = output_size;
	ctx->wx = borrowTensor(wx_dims, wx_shape);
	if(!ctx->wx){
		freeDenseContext(ctx);
		assert(0);
		return NULL;
	}

	return ctx;
}

void freeDenseContext(DenseContext_sn* ctx){
	if(!ctx) return;

	releaseTensor(ctx->weights);
	releaseTensor(ctx->bias);
	releaseTensor(ctx->wx);

	free(ctx);
}
