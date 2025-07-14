#include <smolnet/alloc.h>

MemoryPool_sn* getFloatPool(){
	static MemoryPool_sn* pool = NULL;
	if(pool != NULL) return pool;

	pool = createPool(MP_FLOAT_INITIAL_SIZE, I_FLOAT);
	pool->freeItem = destroyFloat;
	return pool;
}

void prepareFloat(void* item){
	(void)item;
}

void destroyFloat(void* item){
	free(item);
}

void releaseFloat(float* f_array){
	MemoryPool_sn* pool = getFloatPool();
	releaseItem(f_array, pool, prepareFloat);
}

float* borrowFloat(int size){
	MemoryPool_sn* pool = getIntPool();
	void** item = borrowItem(pool);
	(void)size;
	if(!item) return NULL;

	if(*item == NULL){
		*item = (void*)calloc(pool->pool_size, sizeof(float)); // lets leave it like this for now.
		pool->item_pool[pool->allocated_size++] = *item;
	}
	return (float*)(*item);
}
