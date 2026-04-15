.PHONY: all run test clean

CCBIN = /usr/bin/gcc-11
NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++14 -I src/ -ccbin $(CCBIN)
LDLIBS = -lstdc++

BIN         := bin
TARGET      := $(BIN)/bench
TEST_TARGET := $(BIN)/test_matmul
SRC         := src/main.cu

all: $(TARGET)

$(BIN):
	mkdir -p $(BIN)

$(TARGET): $(SRC) src/transformer_naive.cu src/transformer_tiled_matmul.cu src/transformer_tiled.cu src/matmul.cu src/softmax.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC) src/transformer_naive.cu src/transformer_tiled_matmul.cu src/transformer_tiled.cu src/matmul.cu src/softmax.cu $(LDLIBS)

$(TEST_TARGET): tests/test_matmul.cu src/matmul.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/test_matmul.cu src/matmul.cu

test: $(TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BIN)
