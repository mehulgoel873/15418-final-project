#pragma once
#include <cuda_runtime.h>

// Shared helper kernels — also used by TransformerSparse.
__global__ void transpose_kernel(float* input, float* output, int rows, int cols);
__global__ void scale_kernel(float* matrix, int N, int d);
__global__ void add_mask_kernel(float* scores, const float* mask, int N);

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
