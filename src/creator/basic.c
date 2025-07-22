#include <smolnet/creator.h>

int* broadcastShape(Tensor_sn* a, Tensor_sn* b, int* dims){
	if(!a || !b || !dims) return NULL;
	*dims = a->dims > b->dims ? a->dims : b->dims;
	int max_dims = *dims;

	int* output_shape = borrowInt(max_dims);
	if(!output_shape) return NULL;

	for(int i = 0; i < max_dims; i++){
		int a_dim = (i >= max_dims - a->dims) ? a->shape[i - (max_dims - a->dims)] : 1;
		int b_dim = (i >= max_dims - b->dims) ? b->shape[i - (max_dims - b->dims)] : 1;

		if(a_dim != b_dim && a_dim != 1 && b_dim != 1){
			releaseInt(output_shape);
			return NULL;
		}

		output_shape[i] = (a_dim > b_dim) ? a_dim : b_dim;
	}

	return output_shape;
}

int getBroadcastIndex(int idx, int original_dim){
	return (original_dim == 1) ? 0 : idx;
}

Tensor_sn* broadcastedTensor(Tensor_sn* a, Tensor_sn* b){
	if(!a || !b) return NULL;

	int max_dims = 0;
	int* shape = broadcastShape(a, b, &max_dims);
	if(!shape || max_dims < 1) return NULL;

	Tensor_sn* result = borrowTensor(max_dims, shape);
	return result;
}

Tensor_sn* addTensors(Tensor_sn* a, Tensor_sn* b){
	if(!a || !b) return NULL;

	Tensor_sn* result = broadcastedTensor(a, b);
	if(!result) return NULL;

	result->creator = borrowCreator(a, b, OP_ADD);
	if(!result->creator){
		releaseTensor(result);
		return NULL;
	}

	int* indecies = borrowInt(result->dims);
	for(int i = 0; i < result->volume; i++){
		int div = 1;

		for(int d = result->dims - 1; d > -1; d--){
			indecies[d] = (i / div) % result->shape[d];
			div *= result->shape[d];
		}

		int a_idx = 0, b_idx = 0;
		int a_stride = 1, b_stride = 1;

		for(int d = a->dims - 1, rd = result->dims - 1; d > -1; d--, rd--){
			int i_a = getBroadcastIndex(indecies[rd], a->shape[d]);
			a_idx += i_a * a_stride;
			a_stride *= a->shape[d];
		}

		for(int d = b->dims - 1, rd = result->dims - 1; d > -1; d--, rd--){
			int i_b = getBroadcastIndex(indecies[rd], b->shape[d]);
			b_idx += i_b * b_stride;
			b_stride *= b->shape[d];
		}

		result->data[i] = a->data[a_idx] + b->data[b_idx];
	}

	releaseInt(indecies);
	
	return result;
}
