#include <smolnet.h>

int main(){
	Tensor* a = tensor_random((int[]){1, 2}, 2);
	Tensor* b = tensor_random((int[]){1, 2}, 2);

	Tensor* c = add_tensors(a, b);
	print_tensor(c);

	free_tensor(c);
	free_tensor(b);
	free_tensor(a);
}
