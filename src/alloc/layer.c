#include <smolnet/alloc.h>

MemoryPool_sn* getLayerPool(){
	static MemoryPool_sn* pool = NULL;
	if(pool != NULL) return pool;

	pool = createPool(MP_LAYER_INITIAL_SIZE, I_LAYER);
	pool->freeItem = destroyLayer;
	return pool;
}

void prepareLayer(void* item){
	Layer_sn* layer = (Layer_sn*)item;
	if(layer) layer->erase(layer);
}


void destroyLayer(void* item){
	Layer_sn* layer = (Layer_sn*)item;
	if(layer) layer->free(layer);
}

void releaseLayer(Layer_sn* layer){
	MemoryPool_sn* pool = getLayerPool();
	releaseItem(layer, pool, prepareLayer);
}

Layer_sn* borrowLayer(int input_size, int output_size){
	MemoryPool_sn* pool = getLayerPool();
	void** item = borrowItem(pool);
	if(!item) return NULL;

	if(*item == NULL){
		*item = (void*)createDenseLayer(input_size, output_size);
		pool->item_pool[pool->allocated_size++] = *item;
	}

	return (Layer_sn*)(*item);
}
