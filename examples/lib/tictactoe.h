#ifndef _TICTACTOE_H_
#define _TICTACTOE_H_

typedef enum { // from X (player 1) perspective
	WIN,
	LOSS,
	DRAW,
	WIN_ILLEGAL,
	LOSS_ILLEGAL
} GameResult;

typedef int (MoveFunction)(double[9], void*);

#include<stdio.h>

int isPositionWinning(double position[9]);
int validMove(double position[9], int move);
int getLegalMoves(double position[9]);
char* resultMessage(GameResult result);
GameResult game(double position[9], MoveFunction* p1, void* p1_state, MoveFunction* p2, void* p2_state);
MoveFunction player;
void printBoard(double position[9]);

#endif
