#pragma once
#include <cuda_runtime.h>

__global__ void transpose_kernel(float* input, float* output, int rows, int cols);
__global__ void scale_kernel(float* matrix, int N, int d);
__global__ void softmax_kernel(float* input, float* output, int row_len, int num_rows);

class TransformerNaive {
public:
    /// Forward pass of the transformer block
    /// q, k, v: input matrices of size N x d
    /// output: output matrix of size N x d
    /// N: number of tokens, d: embedding dimension, d <= N
    void forward(float* q, float* k, float* v, float* output, int N, int d);
};
