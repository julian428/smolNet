#include <lib/network.h>

void freeLayer(Layer l){
	if(!isValidLayer(l)) return;
	for(int i = 0; i < l.output_length; i++) free(l.weights[i]);
	free(l.weights);
	free(l.bias);
	free(l.output);
	free(l.activated_output);
}

int isValidLayer(Layer l){
	if(l.input_length < 1) return 0;
	if(l.output_length < 1) return 0;
	if(l.bias == NULL) return 0;
	if(l.output == NULL) return 0;
	if(l.activated_output == NULL) return 0;
	if(l.weights == NULL) return 0;
	for(int i = 0; i < l.output_length; i++){
		if(l.weights[i] == NULL) return 0;
	}
	return 1;
}

Layer invalidLayer(){
	return (Layer){
		(Activation){},
		NULL,
		NULL,
		NULL,
		NULL,
		-1,
		-1
	};
}

Layer emptyLayer(int input_length, int output_length, Activation activation){
	if(input_length < 1 || output_length < 1) return invalidLayer();
	double** weights = (double**)malloc(output_length * sizeof(double*));
	if(weights == NULL) return invalidLayer();
	

	double* bias = (double*)malloc(output_length * sizeof(double));
	if(bias == NULL){
		free(weights);
		return invalidLayer();
	}

	double* output = (double*)malloc(output_length * sizeof(double));
	if(output == NULL){
		free(weights);
		free(bias);
		return invalidLayer();
	}

	double* activated_output = (double*)malloc(output_length * sizeof(double));
	if(activated_output == NULL){
		free(weights);
		free(bias);
		free(output);
		return invalidLayer();
	}

	for(int i = 0; i < output_length; i++){
		weights[i] = (double*)malloc(input_length * sizeof(double));
		if(weights[i] == NULL){
			for(int j = 0; j < i; j++) free(weights[j]);
			free(weights);
			free(bias);
			free(output);
			free(activated_output);
			return invalidLayer();
		}
	}
	
	return (Layer){
		activation,
		weights,
		bias,
		output,
		activated_output,
		input_length,
		output_length
	};
}

Layer initializeLayer(int input_length, int output_length, Activation activation){
	Layer l = emptyLayer(input_length, output_length, activation);
	if(!isValidLayer(l)) return l;

	static int seeded = 0;
	if(!seeded){
		srand(time(NULL));
	}

	for(int i = 0; i < l.output_length; i++){
		for(int j = 0; j < l.input_length; j++){
			l.weights[i][j] = (double)rand() / RAND_MAX * 2 - 1;
		}
		l.bias[i] = (double)rand() / RAND_MAX * 2 - 1;
	}

	return l;
}

void freeNetwork(Network n){
	if(!isValidNetwork(n)) return;
	for(int i = 0; i < n.length; i++) freeLayer(n.layers[i]);
}

int isValidNetwork(Network n){
	if(n.layers == NULL) return 0;
	if(n.length < 1) return 0;
	if(n.input_length < 1) return 0;
	if(n.output_length < 1) return 0;
	if(n.learning_rate <= 0) return 0;
	for(int i = 0; i < n.length; i++){
		if(!isValidLayer(n.layers[i])) return 0;
	}
	return 1;
}

Network invalidNetwork(){
	return (Network){NULL, (Cost){}, -1, -1, -1, 0};
}

Network emptyNetwork(int length, double learning_rate, Cost cost){
	if(length < 1 || learning_rate <= 0) return invalidNetwork();
	int network_length = length - 1; // count of the weight matrices between layers
	
	Layer* layers = (Layer*)malloc(network_length * sizeof(Layer));
	if(layers == NULL) return invalidNetwork();

	return (Network){
		layers,
		cost,
		network_length,
		0,
		0,
		learning_rate
	};
}

Network initializeNetwork(int* layer_sizes, int length, double learning_rate, Activation activation, Activation output_activation, Cost cost){
	Network n = emptyNetwork(length, learning_rate, cost);
	if(learning_rate <= 0){
		perror("invalid learning rate.\n");
		freeNetwork(n);
		return invalidNetwork();
	}

	n.input_length = layer_sizes[0];
	n.output_length = layer_sizes[length-1];

	for(int i = 0; i < n.length; i++){
		if(i == n.length - 1)
			n.layers[i] = initializeLayer(layer_sizes[i], layer_sizes[i+1], output_activation);
		else
			n.layers[i] = initializeLayer(layer_sizes[i], layer_sizes[i+1], activation);
	}
	
	return n;
}

void calculateOutput(Layer l, double* input){
	for(int i = 0; i < l.output_length; i++){
		double sum = 0;
		for(int j = 0; j < l.input_length; j++){
			sum += l.weights[i][j] * input[j];
		}
		l.output[i] = sum + l.bias[i];
		l.activated_output[i] = l.activation.func(l.output[i]);
	}
}

void calculateNetworkOutputs(Network n, double* input){
	calculateOutput(n.layers[0], input);
	for(int i = 1; i < n.length; i++){
		calculateOutput(n.layers[i], n.layers[i-1].activated_output);
		if(!hasSingularActivation(n.layers[i].activation)){
			n.layers[i].activation.apply(n.layers[i].activated_output, n.layers[i].output_length);
		}
	}
}

void backPropagateNetwork(Network n, double* input, double* expected){
	if(!isValidNetwork(n)) return;
	calculateNetworkOutputs(n, input);
	
	// allocating memory for gradients
	double** error_signal_gradients = (double**)malloc(n.length * sizeof(double*)); // the same as bias gradient
	double*** weight_gradients = (double***)malloc(n.length * sizeof(double**));

	for(int i = 0; i < n.length; i++){
		Layer current_layer = n.layers[n.length-i-1];
		error_signal_gradients[i] = (double*)malloc(current_layer.output_length * sizeof(double));
		weight_gradients[i] = (double**)malloc(current_layer.output_length * sizeof(double*));
		for(int j = 0; j < current_layer.output_length; j++){
			weight_gradients[i][j] = (double*)malloc(current_layer.input_length * sizeof(double));
		}
	}

	// calculating gradients
	// the gradients to layer order has been reversed. So the gradient for the last layer is the first
	
	// signal error gradients
	// first for the last layer because of the cost function
	for(int i = 0; i < n.output_length; i++){
		error_signal_gradients[0][i] = n.cost.derivative(n.layers[n.length-1].activated_output[i], expected[i]); // cross entropy derivative for softmax so its simpler
	}

	for(int i = 1; i < n.length; i++){
		// this is calculating the product of the previous error signal gradient and a transposed weight matrix of the current layer. This is why the output and input lengths are switched.
		Layer current_layer = n.layers[n.length-i-1];
		Layer next_layer = n.layers[n.length - i]; // by the next layer I mean the next layer in the network, but the previous, index wise.
		for(int j = 0; j < next_layer.input_length; j++){
			error_signal_gradients[i][j] = 0;
			for(int k = 0; k < next_layer.output_length; k++){
				error_signal_gradients[i][j] += next_layer.weights[k][j] * error_signal_gradients[i-1][k];
			}
			error_signal_gradients[i][j] *= current_layer.activation.derivative(current_layer.output[j]);
		}
	}


	// weights gradients
	
	for(int i = 0; i < n.length-1; i++){
		Layer current_layer = n.layers[n.length - i - 1];
		Layer previous_layer = n.layers[n.length - i - 2]; // by the previous layer I mean the previous layer in the network, but the next, index wise.
		for(int j = 0; j < current_layer.input_length; j++){
			for(int k = 0; k < current_layer.output_length; k++){
				weight_gradients[i][k][j] = previous_layer.activated_output[j] * error_signal_gradients[i][k]; 
			}
		}
	}

	for(int i = 0; i < n.input_length; i++){
		for(int j = 0; j < n.layers[0].output_length; j++){
			weight_gradients[n.length-1][j][i] = input[i] * error_signal_gradients[n.length-1][j];
		}
	}


	// updating parameters
	
	// updating bias-es
	
	for(int i = 0; i < n.length; i++){
		Layer current_layer = n.layers[i];
		for(int j = 0; j < current_layer.output_length; j++){
			current_layer.bias[j] -= n.learning_rate * error_signal_gradients[n.length - i - 1][j];
		}
	}

	// updating weights
	
	for(int i = 0; i < n.length; i++){
		Layer current_layer = n.layers[i];
		for(int j = 0; j < current_layer.output_length; j++){
			for(int k = 0; k < current_layer.input_length; k++){
				current_layer.weights[j][k] -= n.learning_rate * weight_gradients[n.length - i - 1][j][k];
			}
		}
	}

	
	// clean up
	for(int i = 0; i < n.length; i++){
		Layer current_layer = n.layers[n.length-i-1];
		for(int j = 0; j < current_layer.output_length; j++){
			free(weight_gradients[i][j]);
		}
		free(weight_gradients[i]);
		free(error_signal_gradients[i]);
	}
	free(weight_gradients);
	free(error_signal_gradients);
}
