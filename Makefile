CC ?= cc
CFLAGS := -std=c99 -O2 -Wall -Wextra -Iinclude
LDLIBS := -lm
SRC := src/rng.c src/tokenizer.c src/model.c src/train.c src/generate.c src/viz.c

.PHONY: all test run clean

all: bin/minigpt bin/tests

bin/minigpt: src/main.c $(SRC)
	@mkdir -p bin results
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

bin/tests: tests/test_minigpt.c $(SRC)
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: bin/tests
	./bin/tests

run: bin/minigpt
	./bin/minigpt all

clean:
	rm -rf bin results/*.svg results/*.csv results/*.txt results/*.bin
