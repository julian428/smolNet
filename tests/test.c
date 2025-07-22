#include <smolnet.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>



int main(){
	Layer_sn* l = borrowLayer(1, 3, 3);

	int t_dims = 2;
	int* t_shape = borrowInt(t_dims);
	t_shape[0] = 3;
	t_shape[1] = 3;
	int t1_dims = 3;
	int* t1_shape = borrowInt(t1_dims);
	t1_shape[0] = 2;
	t1_shape[1] = 1;
	t1_shape[2] = 3;

	srand(time(NULL));
	Tensor_sn* t = borrowTensor(t_dims, t_shape);
	for(int i = 0; i < t->volume; i++){
		t->data[i] = (float)rand() / INT_MAX;
	}
	Tensor_sn* t1 = borrowTensor(t1_dims, t1_shape);
	for(int i = 0; i < t1->volume; i++){
		t1->data[i] = (float)rand() / INT_MAX;
	}

	Tensor_sn* t2 = addTensors(t, t1);
	(void)l;
	printTensor(t);
	printf("\n");
	printTensor(t1);
	printf("\n");
	printTensor(t2);
	printf("\n");
	printCreator(t2->creator);
}
