.PHONY: all run clean

NVCC      := nvcc
NVCCFLAGS := -O2 -std=c++14 -I src/

TARGET := transformer_naive
SRC    := src/main.cu

all: $(TARGET)

$(TARGET): $(SRC) src/transformer_naive.cu src/matmul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf build
