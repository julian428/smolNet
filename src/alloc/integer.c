#include <smolnet/alloc.h>

MemoryPool_sn* getIntPool(){
	static MemoryPool_sn* pool = NULL;
	if(pool != NULL) return pool;

	pool = createPool(MP_INTEGER_INITIAL_SIZE, I_INTEGER);
	pool->freeItem = destroyInt;
	return pool;
}

void prepareInt(void* item){
	(void)item;
}

void destroyInt(void* item){
	free(item);
}

void releaseInt(int* n_array){
	MemoryPool_sn* pool = getIntPool();
	releaseItem(n_array, pool, prepareInt);
}

int* borrowInt(int size){
	MemoryPool_sn* pool = getIntPool();
	void** item = borrowItem(pool);
	(void)size;
	if(item == NULL) return NULL;

	if(*item == NULL){
		*item = (void*)calloc(pool->pool_size, sizeof(int)); // lets leave it like this for now.
		pool->item_pool[pool->allocated_size++] = *item;
	}

	return (int*)(*item);
}
