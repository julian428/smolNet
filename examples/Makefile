CC = gcc
CFLAGS = -I../include -Ilib -Wall -Wextra -lm -g -fsanitize=address
LDFLAGS = -L../build -lsmolnet

EXAMPLES = tictactoe

.PHONY: all clean

tictactoe: tictactoe_example.c lib/tictactoe.c
	mkdir -p build
	$(CC) $(CFLAGS) tictactoe_example.c lib/tictactoe.c -o build/tictactoe_example $(LDFLAGS)

doors: doors_example.c
	mkdir -p build
	$(CC) $(CFLAGS) doors_example.c -o build/doors_example $(LDFLAGS)

clean:
	rm -rf build

