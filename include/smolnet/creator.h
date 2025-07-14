#ifndef _CREATOR_H_
#define _CREATOR_H_

#include <stdlib.h>
#include <assert.h>

typedef struct Tensor Tensor_sn;

typedef void (backwardFunction)(Tensor_sn*);
typedef enum {
	OP_NONE,
	OP_ADD
} CreatorType;

typedef struct Creator {
	Tensor_sn* dad;
	Tensor_sn* mom;
	
	backwardFunction* back;

	CreatorType type;
} Creator_sn;

Creator_sn* createCreator();

#endif
