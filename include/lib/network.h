#ifndef _NETWORK_H_

#define _NETWORK_H_

#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<time.h>

#include "activation.h"

typedef struct LayerStruct {
	Activation activation;
	double** weights; // output_length X input_length
	double* bias; // length of output_length
	double* output;
	double* activated_output;
	int input_length; // width of matrix
	int output_length; // height of matrix
} Layer;

typedef struct NetworkStruct {
	Layer* layers;
	Cost cost;
	int length;
	int input_length;
	int output_length;
	double learning_rate;
} Network;


// clean up
void freeLayer(Layer l);
void freeNetwork(Network n);

//helpers
Layer invalidLayer();
int isValidLayer(Layer l);
int compareLayers(Layer l1, Layer l2);
Network invalidNetwork();
int isValidNetwork(Network n);
int compareNetworks(Network n1, Network n2);

// initialiazers
Layer emptyLayer(int input_length, int output_length, Activation activation);
Layer initializeLayer(int input_length, int output_length, Activation activation);
Network emptyNetwork(int length, double learning_rate, Cost cost);
Network initializeNetwork(int* layer_sizes, int length, double learning_rate, Activation activation, Activation output_activation, Cost cost);

// computational
void calculateOutput(Layer l, double* input); // input has to be the length of l.width
void calculateNetworkOutputs(Network n, double* input); // input has to be the length of layers[0].width
void backPropagateNetwork(Network n, double* input, double* expected);

#endif
