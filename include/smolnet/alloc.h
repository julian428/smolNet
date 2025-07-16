#ifndef _ALLOC_H_
#define _ALLOC_H_

#include <assert.h>
#include <stdlib.h>

#include "tensor.h"
#include "layer.h"
#include "creator.h"

constexpr int MP_NETWORK_INITIAL_SIZE = 4;
constexpr int MP_LAYER_INITIAL_SIZE = 16;
constexpr int MP_TENSOR_INITIAL_SIZE = 64;
constexpr int MP_CREATOR_INITIAL_SIZE = 64;
constexpr int MP_INTEGER_INITIAL_SIZE = 64;
constexpr int MP_FLOAT_INITIAL_SIZE = 256;

typedef void (destroyItem)(void*);
typedef void (prepareItem)(void*);

typedef enum {
	I_INTEGER,
	I_FLOAT,
	I_CREATOR,
	I_TENSOR,
	I_LAYER,
	I_NETWORK
} ItemType;

typedef struct FreeList {
	void* item;
	struct FreeList* next;
} FreeList_sn;

typedef struct MemoryPool {
	void** item_pool;
	FreeList_sn* free_list;
	FreeList_sn* free_head;
	destroyItem* freeItem;
	void (*free)(struct MemoryPool*);

	ItemType type;
	int pool_size;
	int free_size;
	int allocated_size;
} MemoryPool_sn;

MemoryPool_sn* createPool(int size, ItemType type);

void destroyPool(MemoryPool_sn* pool);
void releaseItem(void* item, MemoryPool_sn* pool, prepareItem* prepareItem);
void** borrowItem(MemoryPool_sn* pool);

// integer
MemoryPool_sn* getIntPool();

prepareItem prepareInt;
destroyItem destroyInt;
void releaseInt(int* n_array);
int* borrowInt(int size);

// float
MemoryPool_sn* getFloatPool();

destroyItem destroyFloat;
void releaseFloat(float* f_array);
float* borrowFloat(int size);

// creator
MemoryPool_sn* getCreatorPool();

destroyItem destroyCreator;
void releaseCreator(Creator_sn* creator);
Creator_sn* borrowCreator();

// tensor
MemoryPool_sn* getTensorPool();

destroyItem destroyTensor;
void releaseTensor(Tensor_sn* tensor);
Tensor_sn* borrowTensor(int dims, ...);

// layer
MemoryPool_sn* getLayerPool();

destroyItem destroyLayer;
void releaseLayer(Layer_sn* layer);
Layer_sn* borrowLayer(int batch_count, int input_size, int output_size);

// network

#endif
