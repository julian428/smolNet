#include <smolnet/alloc.h>

MemoryPool_sn* getTensorPool(){
	static MemoryPool_sn* pool = NULL;
	if(pool != NULL) return pool;

	pool = createPool(MP_TENSOR_INITIAL_SIZE, I_TENSOR);
	pool->freeItem = destroyTensor;
	return pool;
}

void prepareTensor(void* item){
	Tensor_sn* tensor = (Tensor_sn*)item;
	if(tensor) tensor->erase(tensor);
}


void destroyTensor(void* item){
	Tensor_sn* tensor = (Tensor_sn*)item;
	if(tensor) tensor->free(tensor);
}

void releaseTensor(Tensor_sn* tensor){
	MemoryPool_sn* pool = getTensorPool();
	releaseItem(tensor, pool, prepareTensor);
}

Tensor_sn* borrowTensor(int dims, ...){
	MemoryPool_sn* pool = getTensorPool();
	void** item = borrowItem(pool);
	if(!item) return NULL;

	va_list args;
	va_start(args, dims);
	if(*item == NULL){
		*item = (void*)createShapedTensor(dims, args);
		pool->item_pool[pool->allocated_size++] = *item;
	}
	va_end(args);
	return (Tensor_sn*)(*item);
}
