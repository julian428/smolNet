#include <smolnet/tensor.h>

Tensor_sn* createTensor(){
	Tensor_sn* tensor = (Tensor_sn*)malloc(sizeof(Tensor_sn));
	if(!tensor){
		assert(0);
		return NULL;
	}

	tensor->free = freeTensor;
	tensor->erase = eraseTensor;

	tensor->dims = 0;
	tensor->volume = 0;
	tensor->batches = 0;
	tensor->batch_size = 0;
	tensor->visited = 0;

	tensor->creator = NULL;
	tensor->data = NULL;
	tensor->grad = NULL;
	tensor->shape = NULL;

	return tensor;
}

Tensor_sn* createShapedTensor(int dims, ...){
	if(dims < 1){
		assert(0);
		return NULL;
	}

	Tensor_sn* tensor = createTensor();
	if(!tensor) return NULL;

	int* shape = borrowInt(dims);
	if(!shape){
		releaseTensor(tensor);
		return NULL;
	}

	tensor->dims = dims;
	tensor->batch_size = 1;
	tensor->volume = 1;
	
	va_list args;
	va_start(args, dims);
	for(int i = 0; i < dims; i++){
		int dim = va_arg(args, int);
		shape[i] = dim;
		tensor->volume *= dim;
		if(i > 0) tensor->batch_size *= dim;
	}
	va_end(args);

	tensor->batches = shape[0];
	tensor->shape = shape;

	return tensor;
}

void freeTensor(Tensor_sn* tensor){
	if(!tensor){
		assert(0);
		return;
	}

	eraseTensor(tensor);
	free(tensor);
}

void eraseTensor(Tensor_sn* tensor){
	if(!tensor){
		assert(0);
		return;
	}

	releaseCreator(tensor->creator);
	releaseFloat(tensor->data);
	releaseFloat(tensor->grad);
	releaseInt(tensor->shape);

	tensor->dims = 0;
	tensor->volume = 0;
	tensor->batches = 0;
	tensor->batch_size = 0;
	tensor->visited = 0;
	
	tensor->creator = NULL;
	tensor->data = NULL;
	tensor->grad = NULL;
	tensor->shape = NULL;
}

void printTensor(Tensor_sn* tensor){
	if(!tensor) return;
	
	for(int i = 0; i < tensor->dims - 1; i++){
		printf("%dx", tensor->shape[i]);
	}
	printf("%d\n", tensor->shape[tensor->dims - 1]);
}
