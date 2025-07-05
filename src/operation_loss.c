#include <lib/operation.h>

// mse
void mse_tensors(Tensor* output, Tensor* expected, Tensor* loss){
	if(!output || !expected || !loss) return;

	if(!similar_tensors(output, expected) || !similar_tensors(expected, loss)){
		fprintf(stderr, "The output tensor and the expected tensor are incompatible for mse.\n");
		exit(EXIT_FAILURE);
	}

	if(is_operation_nesecary(T_MSE, output, expected, loss)){
		if(loss->creator) free_operation(loss->creator);
		loss->creator = create_operation(T_MSE, output, expected);
	}

	for(int i = 0; i < loss->hyper_volume; i++){
		double diff = output->data[i] - expected->data[i];
		loss->data[i] = diff * diff;
	}
}

void backward_mse_tensor(Tensor* self){
	if(!self || !self->creator || !self->grad) return;

	Tensor* output = self->creator->input1;
	Tensor* expected = self->creator->input2;
	if(!output || !output->grad || !output->data) return;
	if(!expected || !expected->data) return;

	for(int i = 0; i < output->hyper_volume; i++){
		output->grad[i] += self->grad[i] * 2 * (output->data[i] - expected->data[i]);
	}
}

// cross entropy
void cel_tensors(Tensor* output, Tensor* expected, Tensor* loss){
	if(!output || !expected || !loss) return;

	if(!similar_tensors(output, expected)){
		fprintf(stderr, "The output tensor and the expected tensor are incompatible for cel.\n");
		exit(EXIT_FAILURE);
	}

	if(is_operation_nesecary(T_CEL, output, expected, loss)){
		if(loss->creator) free_operation(loss->creator);
		loss->creator = create_operation(T_CEL, output, expected);
	}

	int sample_length = output->shape[output->dims - 1];
	int batch_size = output->hyper_volume / sample_length;
	double total_loss = 0.0;

	for(int i = 0; i < batch_size; i++){
		for(int j = 0; j < sample_length; j++){
			int index = i * sample_length + j;
			double y = output->data[index];
			double e = expected->data[index];

			y = fmax(y, EPSILON);
			total_loss += -e * log(y);
		}
	}

	loss->data[0] = total_loss / batch_size;
}

void backward_cel_tensor(Tensor* self){
	if(!self || !self->creator || !self->grad) return;

	Tensor* output = self->creator->input1;
	Tensor* expected = self->creator->input2;
	if(!output || !output->grad || !output->data) return;
	if(!expected || !expected->data) return;

	if(output->creator->type != T_SOFTMAX){
		fprintf(stderr, "Softmax has to be the activation layer for Cross Entropy Loss (cel).\n");
	}

	int sample_length = output->shape[output->dims - 1];
	int batch_size = output->hyper_volume / sample_length;

	// assumes softmax;
	for(int i = 0; i < output->hyper_volume; i++){
		double y = output->data[i];
		double e = expected->data[i];
		output->grad[i] += (y - e) / batch_size;
	}
}
