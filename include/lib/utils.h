#ifndef _UTILS_H_
#define _UTILS_H_

#include<stdio.h> // temporary

typedef int (mat_indexer)(int row, int col, int width, int height);

int aggregate_multiply(int* array, int length);

mat_indexer index_regular;
mat_indexer index_transposed;
void mat_mul(double* a, double* b, double* c, int a_width, int a_height, int b_width, mat_indexer* index_a, mat_indexer* index_b);

#endif
