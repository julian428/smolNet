#ifndef _UTILS_H_
#define _UTILS_H_

#include<math.h>
#include<stdio.h>
#include<stdlib.h>

typedef int (mat_indexer)(int row, int col, int width, int height);

int aggregate_multiply(int* array, int length);
int equal_vectors(int* array1, int length1, int* array2, int length2);

mat_indexer index_regular;
mat_indexer index_transposed;
void mat_mul(double* a, double* b, double* c, int a_width, int a_height, int b_width, mat_indexer* index_a, mat_indexer* index_b, int do_sum);
void softmax(double* input, double* output, int length);

void die(void* function);

#endif
