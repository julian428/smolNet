#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<math.h>
#include<float.h>

#include <smolnet.h>
#include "tictactoe.h"

#define LEARN_ITERATIONS 1000000

static Network bot;

int maxIndex(double* vector, int length){
	int index = 0;
	double value = vector[0];
	for(int i = 1; i < length; i++){
		if(value > vector[i]) continue;
		value = vector[i];
		index = i;
	}

	return index;
}

int botMove(double position[9]){
	calculateNetworkOutputs(bot, position);
	double* output = bot.layers[bot.length-1].activated_output;
	softmax(output, bot.output_length);

	double max_probability = 0;
	int move = -1;
	for(int i = 0; i < 9; i++){
		if(position[i] != 0) continue;
		if(output[i] < max_probability) continue;
		max_probability = output[i];
		move = i;
	}

	return move;
}

void learn(){
	double board[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	GameResult result = game(board, botMove, botMove);
	
	int last_move = maxIndex(bot.layers[bot.length-1].activated_output, bot.output_length);
	int legal_moves = getLegalMoves(board);
	double expected[9] = {0};
	
	switch(result){
		case WIN:
			board[last_move] = 0;
		case WIN_ILLEGAL:
		case LOSS_ILLEGAL:
			for(int i = 0; i < 9; i++){
				if(i == last_move || board[i] != 0) continue;
				expected[i] = 1.0 / legal_moves;
			}
			break;
		case LOSS:
			board[last_move] = 0;
			expected[last_move] = 1.0;
			break;
		case DRAW:
			board[last_move] = 0;
			for(int i = 0; i < 9; i++){
				if(board[i] != 0) continue;
				expected[i] = 1.0 / (legal_moves + 1);
			}
			break;
	}

	//printf("loss: %lf\n", cel().eval(bot.layers[bot.length-1].activated_output, expected, bot.output_length));

	backPropagateNetwork(bot, board, expected);
}


int main(){
	char* network_filename = "network.sn";
	
	bot = readNetwork(network_filename);
	if(!isValidNetwork(bot)){
		bot = initializeNetwork((int[]){9, 12, 18, 24, 30, 24, 18, 12, 9}, 9, 0.01, relu(), softmax(), cel());
	}
	if(!isValidNetwork(bot)) {
		perror("The network is invalid.\n");
		return 1;
	}

	printNetwork(bot);
	
	for(int i = 0; i < LEARN_ITERATIONS; i++) learn();

	// play O with the user
	double board[9] = {0};
	GameResult result1 = game(board, player, botMove);
	printBoard(board);
	printf("%s\n", resultMessage(result1));

	// play with X
	double board2[9] = {0};
	GameResult result2 = game(board2, botMove, player);
	printBoard(board2);
	printf("%s\n", resultMessage(result2));

	saveNetwork(bot, network_filename);
	freeNetwork(bot);
}
