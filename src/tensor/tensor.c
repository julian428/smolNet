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

Tensor_sn* createShapedTensor(int dims, int* shape){
	if(!shape || dims < 1){
		assert(0);
		return NULL;
	}

	Tensor_sn* tensor = createTensor();
	if(!tensor) return NULL;

	tensor->dims = dims;
	tensor->batch_size = 1;
	tensor->volume = 1;
	
	for(int i = 0; i < dims; i++){
		tensor->volume *= shape[i];
		if(i > 0) tensor->batch_size *= shape[i];
	}

	tensor->batches = shape[0];
	tensor->shape = shape;

	float* data = borrowFloat(tensor->volume);
	if(!data){
		tensor->erase(tensor);
		releaseTensor(tensor);
		return NULL;
	}
	tensor->data = data;

	float* grad = borrowFloat(tensor->volume);
	if(!grad){
		tensor->erase(tensor);
		releaseTensor(tensor);
		return NULL;
	}
	tensor->grad = grad;

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
	
	printf("%p, ", tensor);
	for(int i = 0; i < tensor->dims - 1; i++){
		printf("%dx", tensor->shape[i]);
	}
	printf("%d\n", tensor->shape[tensor->dims - 1]);

	for(int i =0; i < tensor->batches; i++){
		for(int j = 0; j < tensor->batch_size; j++){
			printf("%+9.5f", tensor->data[tensor->batch_size * i + j]);
		}
		printf("\n");
	}
}
