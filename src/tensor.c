#include <lib/tensor.h>

void free_tensor(Tensor* t){
	if(t == NULL) return;
	printf("%d\t%d\n", t->id, t->ref_count);
	assert(t->ref_count == 0);
	
	if(t->data) free(t->data);
	if(t->grad) free(t->grad);
	if(t->shape) free(t->shape);
	if(t->creator){
		if(t->creator->input1) t->creator->input1->ref_count--;
		if(t->creator->input2) t->creator->input2->ref_count--;
		free_operation(t->creator);
	}
	free(t);
	t = NULL;
}

void zero_gradient(Tensor* t){
	if(!t || !t->grad) return;
	for(int i = 0; i < t->hyper_volume; i++) t->grad[i] = 0;
}

Tensor* tensor_zeros(int* shape, int dims){
	Tensor* t = (Tensor*)malloc(sizeof(Tensor));
	if(t == NULL) return NULL;

	int hyper_volume = aggregate_multiply(shape, dims);

	t->data = (double*)calloc(hyper_volume, sizeof(double));
	if(t->data == NULL){
		free_tensor(t);
		return NULL;
	}

	t->grad = (double*)calloc(hyper_volume, sizeof(double));
	if(t->grad == NULL){
		free_tensor(t);
		return NULL;
	}

	t->shape = (int*)malloc(dims * sizeof(int));
	if(t->shape == NULL){
		free_tensor(t);
		return NULL;
	}

	memcpy(t->shape, shape, dims * sizeof(int));
	t->dims = dims;
	t->hyper_volume = hyper_volume;
	t->batches = shape[0];
	t->batch_size = hyper_volume / shape[0];
	t->visited = 0;
	t->creator = NULL;
	t->owner = NULL;
	t->ref_count = 0;
	static int next_id = 0;
	t->id = next_id++;

	return t;
}

Tensor* tensor_random(int* shape, int dims){
	Tensor* t = tensor_zeros(shape, dims);
	if(!t) return NULL;

	static int seeded = 0;
	if(!seeded){
		srand(time(NULL));
		seeded = 1;
	}

	for(int i = 0; i < t->hyper_volume; i++){
		t->data[i] = ((double)rand() / RAND_MAX) * 2 - 1;
	}

	return t;
}

Tensor* tensor_data(double* data, int* shape, int dims){
	Tensor* t = tensor_zeros(shape, dims);
	if(!t) return NULL;

	for(int i = 0; i < t->hyper_volume; i++){
		t->data[i] = data[i];
	}

	return t;
}

void print_tensor(Tensor* t){
	printf("shape: %d", t->shape[0]);
	for(int i = 1; i < t->dims; i++){
		printf(" x %d", t->shape[i]);
	}
	printf("\n");

	printf("creator: %p", t->creator);
	printf("\n\n\n");

	printf("data:\n");
	for(int i = 0; i < t->hyper_volume; i++){
		if(i % t->shape[t->dims-1] == 0) printf("\n");
		printf("%.4lf\t", t->data[i]);
	}
	printf("\n\n\n");

	printf("gradients:\n");
	for(int i = 0; i < t->hyper_volume; i++){
		if(i % t->shape[t->dims-1] == 0) printf("\n");
		printf("%.4lf\t", t->grad[i]);
	}
	printf("\n");;
}
