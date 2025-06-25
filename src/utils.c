#include <lib/utils.h>

void printVector(double* vector, int length){
	for(int i = 0; i < length; i++) printf("%lf\t", vector[i]);
	printf("\n");
}

void printLayer(Layer l){
	for(int i = 0; i < l.output_length; i++){
		printf("%d. | ", i);
		for(int j = 0; j < l.input_length; j++){
			printf("%.2lf ", l.weights[i][j]);
		}
		printf("+ ( %.2lf )\n", l.bias[i]);
	}
}

void printNetwork(Network n){
	for(int i = 0; i < n.length; i++) printf("%d -> ", n.layers[i].input_length);
	printf("%d\n", n.layers[n.length-1].output_length);
}
