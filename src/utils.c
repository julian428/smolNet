#include <lib/utils.h>

int aggregate_multiply(int* array, int length){
	int hyper_volume = 1;
	for(int i = 0; i < length; i++) hyper_volume *= array[i];
	return hyper_volume;
}

int equal_vectors(int* array1, int length1, int* array2, int length2){
	if(length2 != length1) return 0;
	for(int i = 0; i < length1; i++){
		if(array1[i] != array2[i]) return 0;
	}
	return 1;
}

int index_regular(int row, int col, int width, int height){
	(void)height;
	return row * width + col;
}

int index_transposed(int row, int col, int width, int height){
	(void)width;
	return row + col * height;
}

void mat_mul(double* a, double* b, double* c, int a_width, int a_height, int b_width, mat_indexer* index_a, mat_indexer* index_b, int do_sum){
	for(int i = 0; i < b_width; i++){
		for(int j = 0; j < a_height; j++){
			double sum = 0;
			for(int k = 0; k < a_width; k++){
				int i_a = index_a(j, k, a_width, a_height);
				int i_b = index_b(k, i, b_width, a_width);
				sum += a[i_a] * b[i_b];
			}
			int index = j * b_width + i;
			if(do_sum) c[index] += sum;
			else c[index] = sum;
		}
	}
}

void softmax(double* input, double* output, int length){
	double base = 0.0;
	for(int i = 0; i < length; i++) base += exp(input[i]);

	for(int i = 0; i < length; i++){
		output[i] = exp(input[i]) / base;
	}
}

void die(void* function){
	fprintf(stderr, "Died in function %p\n", function);
	exit(EXIT_FAILURE);
}
