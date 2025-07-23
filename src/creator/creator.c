#include <smolnet/creator.h>

Creator_sn* createCreator(Tensor_sn* mom, Tensor_sn* dad, operationFunction* revFunc){
	Creator_sn* creator = malloc(sizeof(Creator_sn));
	if(!creator){
		assert(0);
		return NULL;
	}

	creator->dad = dad;
	creator->mom = mom;
	creator->back = revFunc;

	return creator;
}

void printCreator(Creator_sn* creator){
	if(!creator) return;
	printf("Mom: %p, Dad: %p\n", creator->mom, creator->dad);
	printf("Reverse function: %p\n", creator->back);
}
