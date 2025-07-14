#include <smolnet/creator.h>

Creator_sn* createCreator(){
	Creator_sn* creator = malloc(sizeof(Creator_sn));
	if(!creator){
		assert(0);
		return NULL;
	}

	creator->dad = NULL;
	creator->mom = NULL;
	creator->back = NULL;
	creator->type = OP_NONE;

	return creator;
}
