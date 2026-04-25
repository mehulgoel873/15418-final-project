.PHONY: all run test clean

CCBIN := $(or $(wildcard /usr/bin/gcc-11),$(wildcard /usr/bin/gcc-12),$(shell which gcc))
NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++14 -I src/ -ccbin $(CCBIN)
LDLIBS = -lstdc++ -lm

BIN         := bin
TARGET      := $(BIN)/bench
TEST_TARGET := $(BIN)/test_matmul
SRC         := src/main.cu

all: $(TARGET)

$(BIN):
	mkdir -p $(BIN)

$(TARGET): $(SRC) src/transformer_naive.cu src/transformer_tiled_matmul.cu src/transformer_tiled.cu src/transformer_sparse.cu src/matmul.cu src/softmax.cu src/datastructures/bcsr.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC) src/transformer_naive.cu src/transformer_tiled_matmul.cu src/transformer_tiled.cu src/transformer_sparse.cu src/matmul.cu src/softmax.cu src/datastructures/bcsr.cu $(LDLIBS)

$(TEST_TARGET): tests/test_matmul.cu src/matmul.cu src/datastructures/bcsr.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/test_matmul.cu src/matmul.cu src/datastructures/bcsr.cu $(LDLIBS)

test: $(TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BIN)
