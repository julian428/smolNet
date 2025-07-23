#include <smolnet/layer.h>

Layer_sn* createLayer(){
	Layer_sn* layer = (Layer_sn*)malloc(sizeof(Layer_sn));
	if(!layer){
		assert(0);
		return NULL;
	}

	layer->free = freeLayer;
	layer->erase = eraseLayer;

	layer->type = L_NONE;
	layer->param_count = 0;

	layer->getParameterRef = NULL;
	layer->forward = NULL;
	layer->context = NULL;
	layer->output = NULL;

	return layer;
}

void freeLayer(Layer_sn* layer){
	if(!layer){
		assert(0);
		return;
	}
	
	eraseLayer(layer);
	free(layer);
}

void eraseLayer(Layer_sn* layer){
	if(!layer){
		assert(0);
		return;
	}

	releaseTensor(layer->output);

	layer->free = freeLayer;
	layer->erase = eraseLayer;

	layer->type = L_NONE;
	layer->param_count = 0;

	layer->getParameterRef = NULL;
	layer->forward = NULL;
	layer->context = NULL;
	layer->output = NULL;
}
