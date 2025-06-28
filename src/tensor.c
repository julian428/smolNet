#include <lib/tensor.h>

void free_tensor(Tensor* t){
	if(t == NULL) return;
	
	if(t->data != NULL) free(t->data);
	if(t->grad != NULL) free(t->grad);
	if(t->shape != NULL) free(t->shape);
	if(t->creator != NULL) free_operation(t->creator);
	free(t);
}

Tensor* tensor_zeros(int* shape, int dims){
	Tensor* t = (Tensor*)malloc(sizeof(Tensor));
	if(t == NULL) return NULL;

	int hyper_volume = aggregate_multiply(shape, dims);

	t->data = (double*)calloc(hyper_volume, sizeof(double));
	if(t->data == NULL){
		free(t);
		return NULL;
	}

	t->grad = (double*)calloc(hyper_volume, sizeof(double));
	if(t->grad == NULL){
		free(t->data);
		free(t);
		return NULL;
	}

	t->shape = (int*)malloc(dims * sizeof(int));
	if(t->shape == NULL){
		free(t->grad);
		free(t->data);
		free(t);
		return NULL;
	}

	memcpy(t->shape, shape, dims * sizeof(int));
	t->dims = dims;
	t->hyper_volume = hyper_volume;
	t->creator = NULL;
	t->visited = 0;

	return t;
}

Tensor* tensor_random(int* shape, int dims){
	Tensor* t = tensor_zeros(shape, dims);

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
