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
	Tensor_sn** (*getParameterRef)(Layer_sn*, int index);

	LayerType type;
	int param_count;
} Layer_sn;

Layer_sn* createLayer();

layerReleaseFunction freeLayer;
layerReleaseFunction eraseLayer;

// dense
typedef struct DenseContext {
	Tensor_sn* weights;
	Tensor_sn* bias;
	Tensor_sn* wx;
} DenseContext_sn;

Layer_sn* createDenseLayer(int input_size, int output_size);
propagationFunction forwardDense;

layerReleaseFunction freeDenseLayer;
layerReleaseFunction eraseDenseLayer;

DenseContext_sn* createDenseContext(int input_size, int output_size);
void freeDenseContext(DenseContext_sn* ctx);
Tensor_sn** getDenseParameterRef(Layer_sn* layer, int index);

#endif
