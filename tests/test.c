#include <smolnet.h>
#include <stdio.h>

__attribute__((destructor))
void cleanUp(){
	MemoryPool_sn* creatorPool = getCreatorPool();
	MemoryPool_sn* tensorPool = getTensorPool();
	MemoryPool_sn* floatPool = getFloatPool();
	MemoryPool_sn* intPool = getIntPool();
	creatorPool->free(creatorPool);
	tensorPool->free(tensorPool);
	floatPool->free(floatPool);
	intPool->free(intPool);
}

int main(){
	Tensor_sn* t = borrowTensor(2, 3, 3);
	printTensor(t);
	(void)t;
}
