#include <lib/utils.h>

int aggregate_multiply(int* array, int length){
	int hyper_volume = 1;
	for(int i = 0; i < length; i++) hyper_volume *= array[i];
	return hyper_volume;
}

int index_regular(int row, int col, int width, int height){
	(void)height;
	return row * width + col;
}

int index_transposed(int row, int col, int width, int height){
	(void)width;
	return row + col * height;
}

void mat_mul(double* a, double* b, double* c, int a_width, int a_height, int b_width, mat_indexer* index_a, mat_indexer* index_b){
	for(int i = 0; i < b_width; i++){
		for(int j = 0; j < a_height; j++){
			double sum = 0;
			for(int k = 0; k < a_width; k++){
				int i_a = index_a(j, k, a_width, a_height);
				int i_b = index_b(k, i, b_width, a_width);
				sum += a[i_a] * b[i_b];
			}
			int index = j * b_width + i;
			c[index] = sum;
		}
	}
}

