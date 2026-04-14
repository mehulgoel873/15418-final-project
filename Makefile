.PHONY: all run test clean

NVCC      := nvcc
NVCCFLAGS := -O2 -std=c++14 -I src/

BIN         := bin
TARGET      := $(BIN)/transformer_naive
TEST_TARGET := $(BIN)/test_matmul
SRC         := src/main.cu

all: $(TARGET)

$(BIN):
	mkdir -p $(BIN)

$(TARGET): $(SRC) src/transformer_naive.cu src/matmul.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC) src/matmul.cu

$(TEST_TARGET): tests/test_matmul.cu src/matmul.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/test_matmul.cu src/matmul.cu

test: $(TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BIN)
