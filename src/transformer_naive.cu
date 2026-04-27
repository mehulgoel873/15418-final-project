#include "transformer_naive.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matmul.cuh"
#include "softmax.cuh"

static void rand_init_device_buf(float* d_ptr, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX;
    cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

__global__ void transpose_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

__global__ void scale_kernel(float* matrix, int N, int d) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < N && col_idx < N) {
        matrix[row_idx * N + col_idx] /= sqrtf((float)d);
    }
}

// scores[i] += mask[i] for i in [0, N*N). Mask is {0, -inf} additive.
__global__ void add_mask_kernel(float* scores, const float* mask, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int idx = row * N + col;
        scores[idx] += mask[idx];
    }
}

TransformerNaive::TransformerNaive(int d) : d_W_q(nullptr), d_W_k(nullptr), d_W_v(nullptr), d_dim(d) {
    size_t bytes = (size_t)d * d * sizeof(float);
    cudaMalloc(&d_W_q, bytes);
    cudaMalloc(&d_W_k, bytes);
    cudaMalloc(&d_W_v, bytes);
    rand_init_device_buf(d_W_q, d * d);
    rand_init_device_buf(d_W_k, d * d);
    rand_init_device_buf(d_W_v, d * d);
}

TransformerNaive::~TransformerNaive() {
    cudaFree(d_W_q);
    cudaFree(d_W_k);
    cudaFree(d_W_v);
}

void TransformerNaive::forward(float* x, float* mask, float* output, int N, int d, int /*granularity*/) {
    float *Q, *K, *V, *K_T, *scores, *probs, *attn_out;
    size_t tok_bytes  = (size_t)N * d * sizeof(float);
    size_t attn_bytes = (size_t)N * N * sizeof(float);
    if (cudaMalloc(&Q,        tok_bytes)  != cudaSuccess ||
        cudaMalloc(&K,        tok_bytes)  != cudaSuccess ||
        cudaMalloc(&V,        tok_bytes)  != cudaSuccess ||
        cudaMalloc(&K_T,      tok_bytes)  != cudaSuccess ||
        cudaMalloc(&scores,   attn_bytes) != cudaSuccess ||
        cudaMalloc(&probs,    attn_bytes) != cudaSuccess ||
        cudaMalloc(&attn_out, tok_bytes)  != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: N=%d d=%d requires ~%.1f GB for attn matrices\n",
                N, d, 2.0 * attn_bytes / 1e9);
        return;
    }

    // Project x into Q, K, V using tiled matmul: (N x d) @ (d x d) -> (N x d).
    matmul_tiled(x, d_W_q, Q, N, d, d);
    matmul_tiled(x, d_W_k, K, N, d, d);
    matmul_tiled(x, d_W_v, V, N, d, d);

    // K^T: (N x d) -> (d x N)
    dim3 block16(16, 16);
    dim3 grid_KT((d + 15) / 16, (N + 15) / 16);
    transpose_kernel<<<grid_KT, block16>>>(K, K_T, N, d);

    // scores = Q K^T : (N x d) @ (d x N) -> (N x N)
    matmul_tiled(Q, K_T, scores, N, d, N);

    // scale by 1/sqrt(d)
    dim3 grid_NN((N + 15) / 16, (N + 15) / 16);
    scale_kernel<<<grid_NN, block16>>>(scores, N, d);

    // additive mask (mask in {0, -inf})
    add_mask_kernel<<<grid_NN, block16>>>(scores, mask, N);

    // probs = softmax(scores) row-wise
    softmax_tiled(scores, probs, N, N);

    // attn_out = probs @ V : (N x N) @ (N x d) -> (N x d)
    matmul_tiled(probs, V, attn_out, N, N, d);

    cudaMemcpy(output, attn_out, tok_bytes, cudaMemcpyDeviceToDevice);

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(K_T);
    cudaFree(scores); cudaFree(probs); cudaFree(attn_out);
}
