#include <smolnet/creator.h>

void add(Tensor_sn* a, Tensor_sn* b, Tensor_sn* res, int i_a, int i_b, int i_res){
	if(!res->creator) res->creator = borrowCreator(a, b, backAdd);
	else{
		res->creator->mom = a;
		res->creator->dad = b;
		res->creator->back = backAdd;
	}
	
	res->data[i_res] = a->data[i_a] + b->data[i_b];
}

void backAdd(Tensor_sn* a, Tensor_sn* b, Tensor_sn* res, int i_a, int i_b, int i_res){
	float res_grad = res->grad[i_res];
	a->grad[i_a] += res_grad;
	b->grad[i_b] += res_grad;
}

void mul(Tensor_sn* a, Tensor_sn* b, Tensor_sn* res, int i_a, int i_b, int i_res){
	if(!res->creator) res->creator = borrowCreator(a, b, backMul);
	else {
		res->creator->mom = a;
		res->creator->dad = b;
		res->creator->back = backMul;
	}

	res->data[i_res] = a->data[i_a] * b->data[i_b];
}

void backMul(Tensor_sn* a, Tensor_sn* b, Tensor_sn* res, int i_a, int i_b, int i_res){
	float res_grad = res->grad[i_res];
	a->grad[i_a] += b->data[i_b] * res_grad;
	b->grad[i_b] += a->data[i_a] * res_grad;
}
