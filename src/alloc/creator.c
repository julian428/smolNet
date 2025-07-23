#include <smolnet/alloc.h>

MemoryPool_sn* getCreatorPool(){
	static MemoryPool_sn* pool = NULL;
	if(pool != NULL) return pool;

	pool = createPool(MP_CREATOR_INITIAL_SIZE, I_CREATOR);
	pool->freeItem = destroyCreator;
	return pool;
}

void prepareCreator(void* item){
	Creator_sn* creator = (Creator_sn*)item;
	creator->dad = NULL;
	creator->mom = NULL;
	creator->back = NULL;
}


void destroyCreator(void* item){
	free(item);
}

void releaseCreator(Creator_sn* creator){
	MemoryPool_sn* pool = getCreatorPool();
	releaseItem(creator, pool, prepareCreator);
}

Creator_sn* borrowCreator(Tensor_sn* mom, Tensor_sn* dad, operationFunction* revFunc){
	MemoryPool_sn* pool = getCreatorPool();
	void** item = borrowItem(pool);
	if(!item) return NULL;

	if(*item == NULL){
		*item = (void*)createCreator(mom, dad, revFunc);
		pool->item_pool[pool->allocated_size++] = *item;
	}
	return (Creator_sn*)(*item);
}
