.PHONY: all run test clean

CCBIN     := gcc
NVCC      := nvcc
NVCCFLAGS := -lineinfo -O3 -std=c++14 -I src/ -ccbin $(CCBIN)
LDLIBS = -lstdc++ -lm

BIN         := bin
TARGET      := $(BIN)/bench
TEST_TARGET := $(BIN)/test_matmul
SRC         := src/main.cu

all: $(TARGET)

$(BIN):
	mkdir -p $(BIN)

$(TARGET): $(SRC) src/transformer_naive.cu src/transformer_sparse.cu src/transformer_sparse_2.cu src/matmul.cu src/sddmm.cu src/softmax.cu src/datastructures/bcsr.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC) src/transformer_naive.cu src/transformer_sparse.cu src/transformer_sparse_2.cu src/matmul.cu src/sddmm.cu src/softmax.cu src/datastructures/bcsr.cu $(LDLIBS)

$(TEST_TARGET): tests/test_matmul.cu src/matmul.cu src/sddmm.cu src/datastructures/bcsr.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/test_matmul.cu src/matmul.cu src/sddmm.cu src/datastructures/bcsr.cu $(LDLIBS)

test: $(TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BIN)
