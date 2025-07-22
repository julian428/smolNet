#include <smolnet/alloc.h>

MemoryPool_sn* createPool(int size, ItemType type){
	MemoryPool_sn* pool = malloc(sizeof(MemoryPool_sn));
	if(!pool){
		assert(0);
		return NULL;
	}

	pool->free = destroyPool;
	pool->freeItem = NULL;
	pool->type = type;
	pool->pool_size = size;
	pool->free_size = size;
	pool->allocated_size = 0;

	pool->item_pool = (void**)calloc(pool->pool_size, sizeof(void*));
	if(!pool->item_pool){
		assert(0);
		return NULL;
	}

	pool->free_list = (FreeList_sn*)malloc(pool->pool_size * sizeof(FreeList_sn));
	if(!pool->free_list){
		free(pool->item_pool);
		assert(0);
		return NULL;
	}

	pool->free_head = &pool->free_list[0];

	for(int i = 0; i < pool->pool_size; i++){
		pool->free_list[i].item = pool->item_pool[i];
		if(i == pool->pool_size - 1)
			pool->free_list[i].next = NULL;
		else
			pool->free_list[i].next = &pool->free_list[i + 1];
	}

	return pool;
}

void destroyPool(MemoryPool_sn* pool){
	if(!pool){
		assert(0);
		return;
	}

	if(pool->item_pool)
		for(int i = 0; i < pool->allocated_size; i++)
			if(pool->freeItem) pool->freeItem(pool->item_pool[i]);
	
	free(pool->item_pool);
	free(pool->free_list);
	free(pool);
}

void releaseItem(void* item, MemoryPool_sn* pool, prepareItem* prepareItem){
	if(item == NULL || pool == NULL){
		return;
	}

	if(pool->pool_size == pool->free_size){
		return;
	}

	prepareItem(item);

	FreeList_sn* node = &pool->free_list[pool->free_size++];
	node->item = item;
	node->next = pool->free_head;
	pool->free_head = node;
}

void** borrowItem(MemoryPool_sn* pool){
	if(pool->free_size == 0){
		assert(0);
		return NULL;
	}

	FreeList_sn* node = pool->free_head;
	pool->free_head = node->next;
	pool->free_size--;
	return &(node->item);
}

__attribute__((destructor))
void cleanUp(){
	MemoryPool_sn* layerPool = getLayerPool();
	MemoryPool_sn* tensorPool = getTensorPool();
	MemoryPool_sn* creatorPool = getCreatorPool();
	MemoryPool_sn* floatPool = getFloatPool();
	MemoryPool_sn* intPool = getIntPool();
	layerPool->free(layerPool);
	tensorPool->free(tensorPool);
	creatorPool->free(creatorPool);
	floatPool->free(floatPool);
	intPool->free(intPool);
}
