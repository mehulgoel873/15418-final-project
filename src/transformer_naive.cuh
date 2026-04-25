#pragma once
#include <cuda_runtime.h>

class TransformerNaive {
public:
    // Allocates and randomly initializes W_q, W_k, W_v (each d x d) on device.
    explicit TransformerNaive(int d);
    ~TransformerNaive();

    // x: input embeddings, shape N x d (device).
    // mask: dense N x N additive mask in {0, -INFINITY} (device).
    // output: shape N x d (device).
    // granularity: unused by the naive impl; accepted for ForwardFn uniformity.
    void forward(float* x, float* mask, float* output, int N, int d, int granularity);

private:
    float* d_W_q;
    float* d_W_k;
    float* d_W_v;
    int    d_dim;
};
