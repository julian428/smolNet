CC = gcc
CFLAGS = -Wall -Wextra -Werror -Iinclude -lm -std=c23
DFLAGS = -g -O0 -fsanitize=address -U_FORTIFY_SOURCE
AR = ar
ARFLAGS = rcs


SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
LIB_NAME = smolnet
STATIC_LIB = $(BUILD_DIR)/lib$(LIB_NAME).a

SRC_FILES = $(shell find $(SRC_DIR) -name "*.c")
OBJ_FILES = $(patsubst %.c, $(BUILD_DIR)/%.o, $(subst $(SRC_DIR)/,,$(SRC_FILES)))

BUILD_TYPE ?= release

ifeq ($(BUILD_TYPE), debug)
	CFLAGS += $(DFLAGS)
else
	CFLAGS += -O2
endif

.PHONY: all clean test

all: $(STATIC_LIB)

debug:
	$(MAKE) clean BUILD_TYPE=debug
	$(MAKE) all BUILD_TYPE=debug

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(STATIC_LIB): $(OBJ_FILES) | $(BUILD_DIR)
	$(AR) $(ARFLAGS) $@ $^

test: all
	$(CC) $(CFLAGS) $(DFLAGS) tests/test.c $(STATIC_LIB) -o $(BUILD_DIR)/test
	./$(BUILD_DIR)/test

clean:
	rm -rf $(BUILD_DIR)
