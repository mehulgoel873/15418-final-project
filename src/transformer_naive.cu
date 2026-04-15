#include "transformer_naive.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "matmul.cuh"
#include "softmax.cuh"

__global__ void transpose_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

/// Scaling kernel: scales the input matrix by 1/sqrt(d)
/// matrix: input/output matrix of size N x N, d: embedding dimension
__global__ void scale_kernel(float* matrix, int N, int d) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < N && col_idx < N) {
        int idx = row_idx * N + col_idx;
        matrix[idx] /= sqrtf((float)d);
    }
}


/// Forward pass of the transformer block
/// q, k, v: input matrices of size N x d
/// output: output matrix of size N x d
/// N: number of tokens, d: embedding dimension, d <= N
void TransformerNaive::forward(float* q, float* k, float* v, float* output, int N, int d) {
    // Allocate memory for intermediate results
    float* k_transposed;
    float* attn_scores;
    float* attn_probs;
    float* attn_output;
    size_t attn_bytes = (size_t)N * N * sizeof(float);
    size_t tok_bytes  = (size_t)N * d * sizeof(float);
    if (cudaMalloc(&k_transposed, tok_bytes)  != cudaSuccess ||
        cudaMalloc(&attn_scores,  attn_bytes) != cudaSuccess ||
        cudaMalloc(&attn_probs,   attn_bytes) != cudaSuccess ||
        cudaMalloc(&attn_output,  tok_bytes)  != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: N=%d d=%d requires ~%.1f GB for attn matrices\n",
                N, d, 2.0 * attn_bytes / 1e9);
        return;
    }

    // Compute attention scores: attn_scores = Q x K^T
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    transpose_kernel<<<gridSize, blockSize>>>(k, k_transposed, N, d);
    cudaDeviceSynchronize();
    matmul_naive(q, k_transposed, attn_scores, N, d, N);

    // Scale attention scores
    scale_kernel<<<gridSize, blockSize>>>(attn_scores, N, d);
    cudaDeviceSynchronize();

    // Apply softmax to attention scores
    softmax_naive(attn_scores, attn_probs, N, N);
    cudaDeviceSynchronize();

    // Compute attention output: attn_output = attn_probs x V
    matmul_naive(attn_probs, v, attn_output, N, N, d);
    cudaDeviceSynchronize();

    // Copy the result to the output matrix
    cudaMemcpy(output, attn_output, N * d * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free allocated memory
    cudaFree(k_transposed);
    cudaFree(attn_scores);
    cudaFree(attn_probs);
    cudaFree(attn_output);
}