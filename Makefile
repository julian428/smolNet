CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -O2
AR = ar
ARFLAGS = rcs


SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
LIB_NAME = smolnet
STATIC_LIB = $(BUILD_DIR)/lib$(LIB_NAME).a

SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_FILES))


.PHONY: all clean test

all: $(STATIC_LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(STATIC_LIB): $(OBJ_FILES)
	$(AR) $(ARFLAGS) $@ $^

test: all
	$(CC) $(CFLAGS) test/test.c $(STATIC_LIB) -o $(BUILD_DIR)/test
	./$(BUILD_DIR)/test

clean:
	rm -rf $(BUILD_DIR)
