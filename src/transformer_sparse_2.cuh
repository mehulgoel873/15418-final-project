#pragma once
#include <cuda_runtime.h>
#include "datastructures/bcsr.cuh"

// Variant of TransformerSparse that uses the gather-based SDDMM kernel
// in src/sddmm.cu (4-arg form taking B as N x d, no transpose), instead of
// the matmul.cu SDDMM (6-arg form taking K^T).
class TransformerSparse2 {
public:
    explicit TransformerSparse2(int d);
    ~TransformerSparse2();

    void forward(float* x, float* mask, float* output, int N, int d, int granularity);

private:
    float* d_W_q;
    float* d_W_k;
    float* d_W_v;
    int    d_dim;
};
