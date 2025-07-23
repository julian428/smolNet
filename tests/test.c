#include <smolnet.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

int main(){
	srand(time(NULL));
	Layer_sn* l = borrowLayer(3, 3);

	Tensor_sn* weights = *l->getParameterRef(l, 0);
	for(int i = 0; i < weights->volume; i++){
		weights->data[i] = ((float)rand() / INT_MAX) * 2 - 1;
	}

	Tensor_sn* bias = *l->getParameterRef(l, 1);
	for(int i = 0; i < bias->volume; i++){
		bias->data[i] = ((float)rand() / INT_MAX) * 2 - 1;
	}

	int input_dims = 2;
	int* input_shape = borrowInt(input_dims);
	input_shape[0] = 3;
	input_shape[1] = 3;

	Tensor_sn* input = borrowTensor(input_dims, input_shape);
	for(int i = 0; i < input->volume; i++){
		input->data[i] = ((float)rand() / INT_MAX) * 2 - 1;
	}

	l->forward(l, input);
	printTensor(input);
	printf("\n");
	printTensor(weights);
	printf("\n");
	printTensor(bias);
	printf("\n");
	printTensor(*l->getParameterRef(l, 2));
	printf("\n");
	printTensor(l->output);
}
