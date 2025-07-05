#include <lib/network.h>

Network* network(int initial_capacity, ...){
	if(initial_capacity < 0) return NULL;
	Network* n = (Network*)malloc(sizeof(Network));
	if(!n) return NULL;

	n->layers = (Layer**)malloc((initial_capacity > 1 ? initial_capacity : 2) * sizeof(Layer*));
	if(!n->layers){
		free(n);
		return NULL;
	}

	if(initial_capacity > 0){
		va_list args;
		va_start(args, initial_capacity);
		for(int i = 0; i < initial_capacity; i++){
			n->layers[i] = va_arg(args, Layer*);
		}
		va_end(args);
	}

	n->length = initial_capacity;
	n->capacity = initial_capacity > 1 ? initial_capacity : 2;

	n->output = NULL;
	n->loss = NULL;
	n->graph = NULL;

	return n;
}


int add_layer(Network* n, Layer* l){
	if(!n || !l || !n->layers) return 0;
	if(n->length >= n->capacity){
		if(!network_resize(n)) return 0;
	}

	n->layers[n->length++] = l;
	if(n->graph){
		free_tensorgraph(n->graph);
		n->graph = NULL;
	}
	return 1;
}

int pop_layer(Network* n){
	if(!n || !n->layers || n->length < 1) return 0;
	
	n->layers[n->length--] = NULL;
	if(n->graph){
		free_tensorgraph(n->graph);
		n->graph = NULL;
	}
	return 1;
}

int network_resize(Network* n){
	if(!n) return 0;
	int new_capacity = 2 * n->capacity;
	Layer** new_layers_location = realloc(n->layers, new_capacity * sizeof(Layer*));
	if(!new_layers_location) return 0;
	n->layers = new_layers_location;
	n->capacity = new_capacity;
	return 1;
}

void network_free(Network* n){
	if(!n) return;
	if(n->graph){
		free_tensorgraph(n->graph);
	}

	if(n->layers){
		for(int i = 0; i < n->length; i++){
			Layer* l = n->layers[i];
			if(l) l->free(l);
		}
	}
	if(n->loss) free_tensor(n->loss);
	free(n->layers);
	free(n);
}

void network_forward(Network* n, Tensor* input){
	if(!n || !input || !n->layers || !n->layers[0]) return;

	for(int i = 0; i < n->length; i++ ){
		if(i == 0) (void)n->layers[0]->forward(n->layers[0], input);
		else if(n->layers[i]) (void)n->layers[i]->forward(n->layers[i], n->layers[i-1]->output);
		else break;
	}

	n->output = n->layers[n->length-1]->output;
}

void network_backward(Network* n){
	if(!n) return;

	if(n->graph){
		calculate_gradients(n->graph);
		return;
	}
	else if(n->loss) n->graph = create_tensorgraph(n->loss);
	else if(n->output) n->graph = create_tensorgraph(n->output);
	else return;

	calculate_gradients(n->graph);
}

void network_zero_gradients(Network* n){
	if(!n) return;

	if(n->graph){
		zero_gradients(n->graph);
		if(n->loss){
			for(int i = 0; i < n->loss->hyper_volume; i++) n->loss->grad[i] = 1.0;
		}
		return;
	}
	else if(n->loss) n->graph = create_tensorgraph(n->loss);
	else if(n->output) n->graph = create_tensorgraph(n->output);
	else return;

	zero_gradients(n->graph);
	if(n->loss){
		for(int i = 0; i < n->loss->hyper_volume; i++) n->loss->grad[i] = 1.0;
	}else if(n->output){
		for(int i = 0; i < n->output->hyper_volume; i++) n->output->grad[i] = 1.0;
	}
}

Tensor* network_calculate_loss(Network* n, void (*loss_function)(Tensor*, Tensor*, Tensor*), Tensor* expected){
	if(!n || !n->output || !expected) return NULL;
	if(!twin_tensors(n->loss, expected)){
		if(n->loss) free_tensor(n->loss);
		n->loss = tensor_zeros(expected->shape, expected->dims);
	}
	loss_function(n->output, expected, n->loss);
	return n->loss;
}

void network_step(Network* n, void (*loss_function)(Tensor*, Tensor*, Tensor*), Tensor* input, Tensor* expected){
	if(!n) return;
	network_forward(n, input);
	if(loss_function && expected) network_calculate_loss(n, loss_function, expected);
	network_zero_gradients(n);
	network_backward(n);
}
