#ifndef _NETWORK_H_
#define _NETWORK_H_

typedef struct Layer Layer_sn;
typedef struct Tensor Tensor_sn;
typedef struct Network Network_sn;
typedef struct TensorGraph TensorGraph_sn;
typedef void (networkReleaseFunction)(Network_sn*);

typedef struct Network {
	networkReleaseFunction* free;

	Layer_sn** layers;
	TensorGraph_sn* graph;
	
	Tensor_sn* output;
	Tensor_sn* loss;

	int layer_count;
} Network_sn;

#endif
