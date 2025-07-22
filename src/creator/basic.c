#include <smolnet/creator.h>

int broadcast_shape(Tensor_sn* a, Tensor_sn* b, int* output_shape){
	if(!a || !b) return -1;
	int max_dims = a->dims > b->dims ? a->dims : b->dims;

	for(int i = 0; i < max_dims; i++){
		int a_dim = (i >= max_dims - a->dims) ? a->shape[i - (max_dims - a->dims)] : 1;
		int b_dim = (i >= max_dims - b->dims) ? b->shape[i - (max_dims - b->dims)] : 1;

		if(a_dim != b_dim && a_dim != 1 && b_dim != 1) return -1;

		output_shape[i] = (a_dim > b_dim) ? a_dim : b_dim;
	}

	return max_dims;
}

int get_broadcast_index(int idx, int original_dim){
	return (original_dim == 1) ? 0 : idx;
}

Tensor_sn* add_tensors(Tensor_sn* a, Tensor_sn* b){
	if(!a || !b) return NULL;

	int max_dims = a->dims > b->dims ? a->dims : b->dims;
	if(max_dims < 1) return NULL;

	int* shape = borrowInt(max_dims);
	if(!shape) return NULL;

	(void)broadcast_shape(a, b, shape);

	Tensor_sn* result = borrowTensor(max_dims, shape);
	result->shape = shape;

	result->dims = max_dims;
	result->batches = shape[0];
	result->batch_size = 1;
	result->volume = 1;

	for(int i = 0; i < max_dims; i++){
		if(i > 0) result->batch_size *= shape[i];
		result->volume *= shape[i];
	}

	result->creator = borrowCreator(a, b, OP_ADD);
	if(!result->creator){
		releaseTensor(result);
		return NULL;
	}

	int* indecies = borrowInt(max_dims);
	for(int i = 0; i < result->volume; i++){
		int div = 1;

		for(int d = max_dims - 1; d > -1; d--){
			indecies[d] = (i / div) % shape[d];
			div *= shape[d];
		}

		int a_idx = 0, b_idx = 0;
		int a_stride = 1, b_stride = 1;

		for(int d = a->dims - 1, rd = max_dims - 1; d > -1; d--, rd--){
			int i_a = get_broadcast_index(indecies[rd], a->shape[d]);
			a_idx += i_a * a_stride;
			a_stride *= a->shape[d];
		}

		for(int d = b->dims - 1, rd = max_dims - 1; d > -1; d--, rd--){
			int i_b = get_broadcast_index(indecies[rd], b->shape[d]);
			b_idx += i_b * b_stride;
			b_stride *= b->shape[d];
		}

		result->data[i] = a->data[a_idx] + b->data[b_idx];
	}

	releaseInt(indecies);
	
	return result;
}
