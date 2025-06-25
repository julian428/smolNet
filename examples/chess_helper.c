#include <stdio.h>

char* PIECES = "pnbrqkPNBRQK";
int POINTS[] = {-1, -3, -3, -5, -9, 0, 1, 3, 3, 5, 9, 0};

int stringLength(char* string){
	int length = 0;
	while(string[length] != '\0') length++;
	return length;
}

int isInArray(char element, char* array, int array_length){
	for(int i = 0; i < array_length; i++){
		if(element == array[i]) return i;
	}
	return -1;
}

int toNumber1to8(char c){
	int number = (int)c - 48;
	if(number < 9 && number > 0) return number;
	return 0;
}

int isFENValid(char* fen, int length){
	int checkSum = 0;
	int rows = 1;
	for(int i = 0; i < length; i++){
		char c = fen[i];
		if(c == 47){
			if(checkSum != 8) return 0;
			rows++;
			checkSum = 0;
			continue;
		}

		int number = toNumber1to8(c);
		if(number > 0) checkSum += number;
		else if(isInArray(c, PIECES, 12) != -1) checkSum++;
		else return 0;
	}

	return checkSum == 8 && rows == 8;
}

int FENToPoints(char* fen, int length){
	int points = 0;
	for(int i = 0; i < length; i++){
		char c = fen[i];
		int array_index = isInArray(c, PIECES, length);
		if(array_index == -1) continue;
		points += POINTS[array_index];
	}

	return points;
}

int main(int argc, char** argv){
	if(argc < 2) return 1;
	char* fen = argv[1];
	int length = stringLength(fen);
	int is_valid = isFENValid(fen, length);

	if(is_valid){
		int points = FENToPoints(fen, length);
		printf("%d\n", points);
	}
}
