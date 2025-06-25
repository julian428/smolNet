#include <lib/activation.h>

// mse

double mseFunc(double x, double y){
	double diff = x - y;
	return diff * diff;
}

double mseDerivative(double x, double y){
	return 2 * (x - y);
}

double mseEval(double* output, double* expected, int length){
	double cost = 0;
	for(int i = 0; i < length; i++)	cost += mseFunc(output[i], expected[i]);

	return cost;
}

Cost mse(){
	return (Cost){COST_MSE, mseFunc, mseDerivative, mseEval};
}

// cel

double celFunc(double output, double expected){
	if(expected < EPSILON) return 0;
	output = fmax(output, EPSILON);
	return -expected * log(output);
}

double celDerivative(double output, double expected){
	return output - expected;
}

double celEval(double* output, double* expected, int length){
	double loss = 0.0;
	for(int i = 0; i < length; i++){
		loss += celFunc(output[i], expected[i]);
	}

	return loss;
}

Cost cel(){
	return (Cost){COST_CEL, celFunc, celDerivative, celEval};
}

// sigmoid

double sigmoidFunc(double x){
	double base = 1 + exp(-x);
	return 1 / base;
}

double sigmoidDerivative(double x){
	double s = sigmoidFunc(x);
	return s * (1.0 - s);
}

void applySigmoid(double* output, int length){
	apply(sigmoidFunc, output, length);
}

void applySigmoidDerivative(double* output, int length){
	apply(sigmoidDerivative, output, length);
}

Activation sigmoid(){
	return (Activation){ACTIVATION_SIGMOID, sigmoidFunc, sigmoidDerivative, applySigmoid, applySigmoidDerivative};
}

// relu

double reluFunc(double x){
	return fmax(0.0, x);
}

double reluDerivative(double x){
	if(x > 0) return 1.0;
	else return 0.0;
}

void applyRelu(double* output, int length){
	apply(reluFunc, output, length);
}

void applyReluDerivative(double* output, int length){
	apply(reluDerivative, output, length);
}

Activation relu(){
	return (Activation){ACTIVATION_RELU, reluFunc, reluDerivative, applyRelu, applyReluDerivative};
}

// leaky relu

double leakyReluFunc(double x){
	return fmax(0.1*x, x);
}

double leakyReluDerivative(double x){
	if(x > 0) return 1.0;
	else return 0.1;
}

void applyLeakyRelu(double* output, int length){
	apply(leakyReluFunc, output, length);
}

void applyLeakyReluDerivative(double* output, int length){
	apply(leakyReluDerivative, output, length);
}

Activation leakyRelu(){
	return (Activation){ACTIVATION_LRELU, leakyReluFunc, leakyReluDerivative, applyLeakyRelu, applyLeakyReluDerivative};
}

// plain

double plainFunc(double x){
	return x;
}

double plainDerivative(double x){
	return 1.0;
}

void applyPlain(double* output, int length){
	apply(plainFunc, output, length);
}

void applyPlainDerivative(double* output, int length){
	apply(plainDerivative, output, length);
}

Activation plain(){
	return (Activation){ACTIVATION_PLAIN, plainFunc, plainDerivative, applyPlain, applyPlainDerivative};
}

// softmax

double softmaxFunc(double x){
	return x;
}

double softmaxDerivative(double x){
	perror("FATAL: softmax should not be useed as a activation function in hidden layers.");
	exit(EXIT_FAILURE);
	return x;
}

void applySoftmax(double* output, int length){
	double sum = 0;
	for(int i = 0; i < length; i++){
		sum += exp(output[i]);
	}

	for(int i = 0; i < length; i++){
		output[i] = exp(output[i]) / sum;
	}
}

void applySoftmaxDerivative(double* output, int length){}

Activation softmax(){
	return (Activation){
		ACTIVATION_SOFTMAX,
		softmaxFunc,
		softmaxDerivative,
		applySoftmax,
		applySoftmaxDerivative
	};
}

// getters

Activation getActivationByID(ActivationID id){
	switch(id){
		case ACTIVATION_SIGMOID:
			return sigmoid();
		case ACTIVATION_RELU:
			return relu();
		case ACTIVATION_LRELU:
			return leakyRelu();
		case ACTIVATION_PLAIN:
		default:
			return plain();
	}
}

Cost getCostByID(CostID id){
	switch(id){
		case COST_CEL:
			return cel();
		case COST_MSE:
		default:
			return mse();
	}
}

//helpers

void apply(ActivationFunction* f, double* vector, int length){
	for(int i = 0; i < length; i++){
		vector[i] = f(vector[i]);
	}
}

int hasSingularActivation(Activation a){
	switch(a.id){
		case ACTIVATION_SOFTMAX:
			return 0;
		default:
			return 1;
	}
}


