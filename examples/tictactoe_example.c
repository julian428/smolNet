#include <smolnet.h>
#include <assert.h>
#include <float.h>
#include "lib/tictactoe.h"

#define EPOCHS 10

static int last_X = -1;
static int last_O = -1;

int best_move(double position[9], double probabilities[9]){
	int move = 0;
	double probability = 0.0;
	for(int i = 0; i < 9; i++){
		if(position[i] != 0) continue;
		if(probability > probabilities[i]) continue;
		probability = probabilities[i];
		move = i;
	}
	return move;
}

int net_move(double position[9], void* state){
	Network* n = (Network*)state;
	assert(n != NULL);

	Tensor* input = tensor_data(position, (int[]){1, 9}, 2);
	network_forward(n, input);

	int move = best_move(position, n->output->data);
	if(last_X == -1 || position[last_X] == 1) last_X = move;
	else last_O = move;

	//free_tensor(input);
	return move;
}

int main(){
	Optimizer* sgd = optimizer_sgd(0.1);
	Network* bot1 = network(
		6,
		dense_layer(9, 18),
		relu_layer(18),
		dense_layer(18, 18),
		relu_layer(18),
		dense_layer(18, 9),
		softmax_layer(9)
	);

	Network* bot2 = network(
		6,
		dense_layer(9, 18),
		relu_layer(18),
		dense_layer(18, 18),
		relu_layer(18),
		dense_layer(18, 9),
		softmax_layer(9)
	);

	for(int i = 0; i < EPOCHS; i++){
		double board[9] = {0};
		GameResult result = game(board, net_move, bot1, net_move, bot2);
		double expected1[9] = {0};
		double expected2[9] = {0};

		if(result == WIN){
			expected1[last_X] = 1;
			expected2[last_X] = 1;
		}
		else if(result == LOSS){
			expected2[last_O] = 1;
			expected1[last_O] = 1;
		}

		Tensor* exX = tensor_data(expected1, (int[]){1, 9}, 2);
		Tensor* exO = tensor_data(expected2, (int[]){1, 9}, 2);

		network_calculate_loss(bot1, cel_tensors, exX);
		network_calculate_loss(bot2, cel_tensors, exO);
		printf("loss X:%7.4lf\tloss O:%7.4lf\n", bot1->loss->data[0], bot2->loss->data[0]);
		network_zero_gradients(bot1);
		network_zero_gradients(bot2);
		network_backward(bot1);
		network_backward(bot2);
		sgd->step(sgd, bot1);
		sgd->step(sgd, bot2);

		free_tensor(exX);
		free_tensor(exO);

		printf("%s\n", resultMessage(result));
	}

	sgd->free(sgd);
	network_free(bot1);
	network_free(bot2);
}
