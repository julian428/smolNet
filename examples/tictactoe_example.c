#include <smolnet.h>

int main(){
	Tensor* input = tensor_random((int[]){1}, 1);
	Tensor* w0 = tensor_random((int[]){1, 2}, 2);
	Tensor* b0 = tensor_random((int[]){1, 2}, 2);
	Tensor* z0 = mul_tensors(input, w0);
	Tensor* zb0 = add_tensors(z0, b0);
	Tensor* w1 = tensor_random((int[]){1, 2}, 2);
	Tensor* b1 = tensor_random((int[]){1, 2}, 2);
	Tensor* z1 = dot_tensors(zb0, w1);
	Tensor* zb1 = add_tensors(z1, b1);

	print_tensor(input);
	printf("\n");

	print_tensor(w0);
	printf("\n");

	print_tensor(b0);
	printf("\n");
	
	print_tensor(z0);
	printf("\n");

	print_tensor(zb0);
	printf("\n");

	print_tensor(w1);
	printf("\n");

	print_tensor(b1);
	printf("\n");
	
	print_tensor(z1);
	printf("\n");

	print_tensor(zb1);
	printf("\n");

	free_tensor(zb1);
	free_tensor(z1);
	free_tensor(b1);
	free_tensor(w1);
	free_tensor(zb0);
	free_tensor(z0);
	free_tensor(b0);
	free_tensor(w0);
	free_tensor(input);
}
