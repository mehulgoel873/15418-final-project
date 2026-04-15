#pragma once
#include <cuda_runtime.h>

class TransformerTiled {
public:
    void forward(float* q, float* k, float* v, float* output, int N, int d);
};
