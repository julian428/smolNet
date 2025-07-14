#ifndef _LAYER_H_
#define _LAYER_H_

#include <stdlib.h>
#include <assert.h>

typedef struct Tensor Tensor_sn;
typedef struct Layer Layer_sn;
typedef void (propagationFunction)(Layer_sn*, Tensor_sn*);
typedef void (layerReleaseFunction)(Layer_sn*);

#include "alloc.h"

typedef enum {
	L_NONE,
	L_DENSE
} LayerType;

typedef struct Layer {
	Tensor_sn* output;
	void* context;
	propagationFunction* forward;
	layerReleaseFunction* free;
	layerReleaseFunction* erase;
	Tensor_sn* (*getParameter)(Layer_sn*);

	LayerType type;
	int param_count;
} Layer_sn;

Layer_sn* createLayer();

layerReleaseFunction freeLayer;
layerReleaseFunction eraseLayer;

#endif
