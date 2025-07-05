#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

typedef struct NetworkS Network;

#include "network.h"

typedef struct OptimizerS Optimizer;

typedef void (OptimizerStep)(Optimizer*, Network*);
typedef void (OptimizerFree)(Optimizer*);

typedef enum {
	O_SGD
} OptimizerType;

typedef struct OptimizerS {
	OptimizerType type;
	double learning_rate;
	OptimizerStep* step;
	OptimizerFree* free;
	void* state;
} Optimizer; 

// SGD
Optimizer* optimizer_sgd(double learning_rate);
OptimizerStep step_sgd;
OptimizerFree free_sgd;

#endif
