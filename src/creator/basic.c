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

void computeStride(int main_index, int secondary_index, int* stride_ref, int* index_ref){
	if(!stride_ref || !index_ref) return;
	*index_ref += getBroadcastIndex(main_index, secondary_index) * (*stride_ref);
	*stride_ref *= secondary_index;
}

Tensor_sn* tensorsOperation(
	Tensor_sn* a, Tensor_sn* b, operationFunction* opFunc
){
	if(!a || !b || !opFunc) return NULL;

	Tensor_sn* result = broadcastedTensor(a, b);
	if(!result) return NULL;

	int indecies[result->dims];
	for(int i = 0; i < result->volume; i++){
		int div = 1;

		for(int d = result->dims - 1; d > -1; d--){
			indecies[d] = (i / div) % result->shape[d];
			div *= result->shape[d];
		}

		int idx[2] = {0};
		int stride[2] = {1};

		int a_dim_offset = result->dims - a->dims;
		int b_dim_offset = result->dims - b->dims;

		for(int d = result->dims - 1; d > -1; d--){
			int da = d - a_dim_offset;
			int db = d - b_dim_offset;

			if(da > -1) computeStride(indecies[d], a->shape[da], &stride[0], &idx[0]);
			if(db > -1) computeStride(indecies[d], b->shape[db], &stride[1], &idx[1]);
		}

		opFunc(a, b, result, idx[0], idx[1], i);
	}
	
	return result;
}
