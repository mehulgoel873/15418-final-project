#pragma once
#include <cuda_runtime.h>
#include "datastructures/bcsr.cuh"

class TransformerSparseSoftmax {
public:
    // Allocates and randomly initializes W_q, W_k, W_v (each d x d) on device.
    explicit TransformerSparseSoftmax(int d);
    ~TransformerSparseSoftmax();

    void forward(float* x, float* mask, float* output, int N, int d, int granularity);

private:
    float* d_W_q;
    float* d_W_k;
    float* d_W_v;
    int    d_dim;
};
