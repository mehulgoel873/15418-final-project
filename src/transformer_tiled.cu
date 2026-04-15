#include "transformer_tiled.cuh"
#include "transformer_naive.cuh"
#include "matmul.cuh"
#include "softmax.cuh"
#include <stdio.h>

void TransformerTiled::forward(float* q, float* k, float* v, float* output, int N, int d) {
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

    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);

    // attn_scores = Q x K^T
    transpose_kernel<<<gridSize, blockSize>>>(k, k_transposed, N, d);
    cudaDeviceSynchronize();
    matmul_tiled(q, k_transposed, attn_scores, N, d, N);

    // TODO: fuse scale and softmax into one kernel to save memory bandwidth
    scale_kernel<<<gridSize, blockSize>>>(attn_scores, N, d);
    cudaDeviceSynchronize();
    softmax_tiled(attn_scores, attn_probs, N, N);
    cudaDeviceSynchronize();

    // attn_output = attn_probs x V
    matmul_tiled(attn_probs, v, attn_output, N, N, d);
    cudaDeviceSynchronize();

    cudaMemcpy(output, attn_output, tok_bytes, cudaMemcpyDeviceToDevice);

    cudaFree(k_transposed);
    cudaFree(attn_scores);
    cudaFree(attn_probs);
    cudaFree(attn_output);
}
