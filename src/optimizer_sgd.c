#include <lib/optimizer.h>

Optimizer* optimizer_sgd(double learning_rate){
	Optimizer* optimizer = (Optimizer*)malloc(sizeof(Optimizer));
	if(!optimizer) return NULL;

	optimizer->type = O_SGD;
	optimizer->learning_rate = learning_rate;
	optimizer->step = step_sgd;
	optimizer->free = free_sgd;
	optimizer->state = NULL;

	return optimizer;
}

void step_sgd(Optimizer* op, Network* n){
	if(!n || !n->layers) return;
	for(int i = 0; i < n->length; i++){
		Layer* l = n->layers[i];
		if(l->param_count < 1 || !l->get_param) continue;
		for(int j = 0; j < l->param_count; j++){
			Tensor* t = l->get_param(l, i);
			if(!t) break;
			for(int k = 0; k < t->hyper_volume; k++) t->data[k] -= op->learning_rate * t->grad[k];
		}
	}
}

void free_sgd(Optimizer* optimizer){
	if(optimizer->type != O_SGD) return;
	free(optimizer);
}
