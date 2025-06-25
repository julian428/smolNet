#ifndef _ACTIVATION_H_

#define _ACTIVATION_H_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define EPSILON 1e-15

typedef double (ActivationFunction)(double);
typedef void (OutputActivationFunction)(double*, int);

typedef double (CostFunction)(double, double);
typedef double (CostFunctionEvaluation)(double*, double*, int);

typedef enum {
	ACTIVATION_SIGMOID,
	ACTIVATION_RELU,
	ACTIVATION_LRELU,
	ACTIVATION_SOFTMAX,
	ACTIVATION_PLAIN
} ActivationID;

typedef enum {
	COST_MSE, // MSE is Mean Square Error
	COST_CEL // CEL is Cross Entropy Loss
} CostID;

typedef struct ActivationStruct {
	ActivationID id;
	ActivationFunction* func;
	ActivationFunction* derivative;
	OutputActivationFunction* apply;
	OutputActivationFunction* applyDerivative;
} Activation;

typedef struct CostStruct {
	CostID id;
	CostFunction* func;
	CostFunction* derivative;
	CostFunctionEvaluation* eval;
} Cost;

// cost functions
CostFunction mseFunc;
CostFunction mseDerivative;
CostFunctionEvaluation mseEval;

CostFunction celFunc;
CostFunction celDerivative;
CostFunctionEvaluation celEval;


// activation functions
ActivationFunction sigmoidFunc;
ActivationFunction sigmoidDerivative;
OutputActivationFunction applySigmoid;
OutputActivationFunction applySigmoidDerivative;

ActivationFunction reluFunc;
ActivationFunction reluDerivative;
OutputActivationFunction applyRelu;
OutputActivationFunction applyReluDerivative;

ActivationFunction leakyReluFunc;
ActivationFunction leakyReluDerivative;
OutputActivationFunction applyLeakyRelu;
OutputActivationFunction applyLeakyReluDerivative;

ActivationFunction plainFunc;
ActivationFunction plainDerivative;
OutputActivationFunction applyPlain;
OutputActivationFunction applyPlainDerivative;

ActivationFunction softmaxFunc;
ActivationFunction softmaxDerivative;
OutputActivationFunction applySoftmax;
OutputActivationFunction applySoftmaxDerivative;

// getter functions
Activation getActivationByID(ActivationID id);
Cost getCostByID(CostID id);

// packaging functions
Activation sigmoid();
Activation relu();
Activation leakyRelu();
Activation softmax();
Activation plain();

Cost mse();
Cost cel();

// helpers
void apply(ActivationFunction* f, double* vector, int length);
int hasSingularActivation(Activation a);

#endif
