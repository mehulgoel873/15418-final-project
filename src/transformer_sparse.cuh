#pragma once
#include <cuda_runtime.h>
#include "datastructures/bcsr.cuh"

class TransformerSparse {
public:
    // Allocates and randomly initializes W_q, W_k, W_v (each d x d) on device.
    explicit TransformerSparse(int d);
    ~TransformerSparse();

    // x: input embeddings, shape N x d (device).
    // mask: dense N x N additive mask in {0, -INFINITY} (device).
    // output: shape N x d (device).
    // granularity: BCSR tile size used for the final probs @ V spmm.
    void forward(float* x, float* mask, float* output, int N, int d, int granularity);

private:
    float* d_W_q;
    float* d_W_k;
    float* d_W_v;
    int    d_dim;
};
