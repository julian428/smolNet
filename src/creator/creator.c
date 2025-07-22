#include <smolnet/creator.h>

Creator_sn* createCreator(Tensor_sn* mom, Tensor_sn* dad, CreatorType type){
	Creator_sn* creator = malloc(sizeof(Creator_sn));
	if(!creator){
		assert(0);
		return NULL;
	}

	creator->dad = dad;
	creator->mom = mom;
	creator->back = NULL;
	creator->type = type;

	return creator;
}

void printCreator(Creator_sn* creator){
	if(!creator) return;
	printf("Mom: %p, Dad: %p\n", creator->mom, creator->dad);

	printf("type: ");
	switch(creator->type){
		case OP_ADD:
		printf("ADDITION\n");
		break;
		case OP_NONE:
		printf("NONE\n");
		break;
		default:
		printf("UNKNOWN\n");
	}
}
