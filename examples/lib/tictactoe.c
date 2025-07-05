#include "tictactoe.h"

int isPositionWinning(double position[9]){
	int Xsum = 0;
	int Osum = 0;
	int winning_positions[8] = {448, 292, 273, 146, 84, 73, 56, 7};

	for(int i = 0; i < 9; i++){
		int val = position[i];
		int ampl = 1 << (8-i);
		if(val == -1) Osum += ampl;
		else if(val == 1) Xsum += ampl;
	}

	for(int i = 0; i < 8; i++){
		if((Xsum & winning_positions[i]) == winning_positions[i]) return 1;
		else if((Osum & winning_positions[i]) == winning_positions[i]) return -1;
	}

	return 0;
}

char* resultMessage(GameResult result){
	switch(result){
		case WIN:
			return "X won!";
		case LOSS:
			return "O won!";
		case DRAW:
			return "Draw :O";
		case WIN_ILLEGAL:
			return "Illegal move by O! >:(";
		case LOSS_ILLEGAL:
			return "Illegal move by X! >:(";
		default:
			return "";
	}
}

int getLegalMoves(double position[9]){
	int legal_moves = 0;
	for(int i = 0; i < 9; i++){
		if(position[i] == 0) legal_moves++;
	}
	return legal_moves;
}

int validMove(double position[9], int move){
	if(move > 8 || move < 0) return 0;
	return position[move] == 0;
}

GameResult game(double position[9], MoveFunction* p1, void* p1_state, MoveFunction* p2, void* p2_state){
	int winning = 0;

	for(int i = 0; (winning = isPositionWinning(position)) == 0 && i < 5; i++){
		int move1 = p1(position, p1_state);
		if(!validMove(position, move1)) return LOSS_ILLEGAL;
		position[move1] = 1;

		if((winning = isPositionWinning(position)) != 0 || i == 4) break;

		int move2 = p2(position, p2_state);
		if(!validMove(position, move2)) return WIN_ILLEGAL;
		position[move2] = -1;
	}

	switch(winning){
		case 1:
			return WIN;
		case -1:
			return LOSS;
		default:
			return DRAW;
	}
}

void printBoard(double position[9]){
	printf("---\n");
	for(int i = 0; i < 9; i++){
		if(i % 3 == 0) printf("\n");
		switch((int)position[i]){
			case 1:
				printf("X ");
				break;
			case -1:
				printf("O ");
				break;
			default:
				printf("  ");
		}
	}
	printf("\n");
}

int player(double position[9], void* state){
	(void)state;
	printBoard(position);

	int move;
	printf("move: ");
	scanf("%d", &move);

	return move;
}
