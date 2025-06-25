#include <lib/storeread.h>

void saveNetwork(Network n, char* file_name){
	FILE* file = fopen(file_name, "wb");

	fwrite(&n.length, sizeof(int), 1, file);
	fwrite(&n.learning_rate, sizeof(double), 1, file);
	fwrite(&n.cost.id, sizeof(CostID), 1, file);

	for(int i = 0; i < n.length; i++){
		Layer l = n.layers[i];

		fwrite(&l.input_length, sizeof(int), 1, file);
		fwrite(&l.output_length, sizeof(int), 1, file);
		fwrite(&l.activation.id, sizeof(ActivationID), 1, file);

		fwrite(l.bias, sizeof(double), l.output_length, file);
		for(int j = 0; j < l.output_length; j++){
			fwrite(l.weights[j], sizeof(double), l.input_length, file);
		}
	}

	fclose(file);
}

Network readNetwork(char* file_name){
	FILE* file = fopen(file_name, "rb");
	if(!file){
		perror("Couldn't open binary network file.\n");
		return invalidNetwork();
	}

	int network_length;
	size_t read_count = fread(&network_length, sizeof(int), 1, file);
	if(read_count != 1){
		perror("Couldn't read the network length properly.\n");
		fclose(file);
		return invalidNetwork();
	}

	double learning_rate;
	read_count = fread(&learning_rate, sizeof(double), 1, file);
	if(read_count != 1){
		perror("Couldn't read the learning rate properly.\n");
		fclose(file);
		return invalidNetwork();
	}

	CostID cost_id;
	read_count = fread(&cost_id, sizeof(CostID), 1, file);
	if(read_count != 1){
		perror("Couldn't read the cost id properly.\n");
		fclose(file);
		return invalidNetwork();
	}

	Cost cost = getCostByID(cost_id);
	Network n = emptyNetwork(network_length+1, learning_rate, cost);

	for(int i = 0; i < network_length; i++){
		int layer_input_length;
		read_count = fread(&layer_input_length, sizeof(int), 1, file);
		if(read_count != 1){
			fprintf(stderr, "Couldn't read layers %d input length properly.\n", i);
			freeNetwork(n);
			fclose(file);
			return invalidNetwork();
		}

		int layer_output_length;
		read_count = fread(&layer_output_length, sizeof(int), 1, file);
		if(read_count != 1){
			fprintf(stderr, "Couldn't read layers %d output length properly.\n", i);
			freeNetwork(n);
			fclose(file);
			return invalidNetwork();
		}

		ActivationID layer_activation_id;
		read_count = fread(&layer_activation_id, sizeof(ActivationID), 1, file);
		if(read_count != 1){
			fprintf(stderr, "Couldn't read layers %d activation id properly.\n", i);
			freeNetwork(n);
			fclose(file);
			return invalidNetwork();
		}

		Activation layer_activation = getActivationByID(layer_activation_id);
		Layer l = emptyLayer(layer_input_length, layer_output_length, layer_activation);

		read_count = fread(l.bias, sizeof(double), layer_output_length, file);
		if(read_count != (size_t)layer_output_length){
			fprintf(stderr, "Couldn't read layers %d bias-es properly. Read %ld, expected %d.\n", i, read_count, layer_output_length);
			freeNetwork(n);
			freeLayer(l);
			fclose(file);
			return invalidNetwork();
		}

		for(int j = 0; j < layer_output_length; j++){
			read_count = fread(l.weights[j], sizeof(double), layer_input_length, file);
			if(read_count != (size_t)layer_input_length){
				fprintf(stderr, "Couldn't read layers %i weight matrix properly. On row %d read %ld, expected %d.\n", i, j, read_count, layer_input_length);
				freeNetwork(n);
				freeLayer(l);
				fclose(file);
				return invalidNetwork();
			}
		}

		n.layers[i] = l;
	}

	n.input_length = n.layers[0].input_length;
	n.output_length = n.layers[network_length-1].output_length;

	fclose(file);
	return n;
}

