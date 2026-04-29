#include "transformer_sparse_softmax.cuh"
#include "transformer_naive.cuh"
#include "matmul.cuh"
#include "softmax.cuh"
#include "timing.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void rand_init_device_buf_softmax(float* d_ptr, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX;
    cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

// Build a granularity-block sparsity mask from the additive softmax mask
static bool* build_tile_mask_from_additive_softmax(const float* d_mask, int N, int granularity) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float* h_mask = (float*)malloc(bytes);
    cudaMemcpy(h_mask, d_mask, bytes, cudaMemcpyDeviceToHost);

    int Tb = N / granularity;
    bool* tile_dense = (bool*)calloc((size_t)Tb * Tb, sizeof(bool));
    for (int bi = 0; bi < Tb; bi++) {
        for (int bj = 0; bj < Tb; bj++) {
            bool any_finite = false;
            for (int ti = 0; ti < granularity && !any_finite; ti++) {
                for (int tj = 0; tj < granularity && !any_finite; tj++) {
                    float v = h_mask[(bi * granularity + ti) * N + (bj * granularity + tj)];
                    if (isfinite(v)) any_finite = true;
                }
            }
            tile_dense[bi * Tb + bj] = any_finite;
        }
    }
    free(h_mask);
    return tile_dense;
}

__global__ void bcsr_to_dense_kernel(BCSRView bcsr, float* dense_out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int T = bcsr.TILING;
        int bi = row / T;
        int bj = col / T;
        int out_idx = bcsr.block_idx[bi * bcsr.num_block_cols + bj];
        
        if (out_idx >= 0) {
            int ti = row % T;
            int tj = col % T;
            int row_start = bcsr.row_ptr[bi];
            int K = bcsr.row_ptr[bi + 1] - row_start;
            int j = out_idx - row_start;
            
            size_t base = (size_t)row_start * T * T;
            dense_out[row * N + col] = bcsr.values[base + (size_t)ti * K * T + (size_t)j * T + tj];
        } else {
            dense_out[row * N + col] = 0.0f;
        }
    }
}

TransformerSparseSoftmax::TransformerSparseSoftmax(int d) : d_W_q(nullptr), d_W_k(nullptr), d_W_v(nullptr), d_dim(d) {
    size_t bytes = (size_t)d * d * sizeof(float);
    cudaMalloc(&d_W_q, bytes);
    cudaMalloc(&d_W_k, bytes);
    cudaMalloc(&d_W_v, bytes);
    rand_init_device_buf_softmax(d_W_q, d * d);
    rand_init_device_buf_softmax(d_W_k, d * d);
    rand_init_device_buf_softmax(d_W_v, d * d);
}

TransformerSparseSoftmax::~TransformerSparseSoftmax() {
    cudaFree(d_W_q);
    cudaFree(d_W_k);
    cudaFree(d_W_v);
}

void TransformerSparseSoftmax::forward(float* x, float* mask, float* output, int N, int d, int granularity) {
    float *Q, *K, *V, *k_transposed;
    float* attn_scores;
    float* attn_probs;
    size_t attn_bytes = (size_t)N * N * sizeof(float);
    size_t tok_bytes  = (size_t)N * d * sizeof(float);
    
    if (cudaMalloc(&Q, tok_bytes) != cudaSuccess ||
        cudaMalloc(&K, tok_bytes) != cudaSuccess ||
        cudaMalloc(&V, tok_bytes) != cudaSuccess ||
        cudaMalloc(&k_transposed, tok_bytes)  != cudaSuccess ||
        cudaMalloc(&attn_scores,  attn_bytes) != cudaSuccess ||
        cudaMalloc(&attn_probs,   attn_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed in TransformerSparseSoftmax\n");
        return;
    }

    // Project x into Q, K, V
    matmul_tiled(x, d_W_q, Q, N, d, d);
    matmul_tiled(x, d_W_k, K, N, d, d);
    matmul_tiled(x, d_W_v, V, N, d, d);

    dim3 blockSize16(16, 16);
    dim3 gridSize16((N + 15) / 16, (N + 15) / 16);

    // 1. Q x K^T -> dense attn_scores
    transpose_kernel<<<gridSize16, blockSize16>>>(K, k_transposed, N, d);
    cudaDeviceSynchronize();
    matmul_tiled(Q, k_transposed, attn_scores, N, d, N);

    // Scale scores
    scale_kernel<<<gridSize16, blockSize16>>>(attn_scores, N, d);
    cudaDeviceSynchronize();

    float* h_attn_scores = (float*)malloc(attn_bytes);
    // Copy the scaled dense scores for BCSR constructor
    cudaMemcpy(h_attn_scores, attn_scores, attn_bytes, cudaMemcpyDeviceToHost);

    // Convert boolean block mask based on additive dense mask
    bool* tile_dense = build_tile_mask_from_additive_softmax(mask, N, granularity);

    // 2. Convert to BCSR & perform sparse softmax
    BCSRMatrix bcsr_scores(h_attn_scores, tile_dense, N, N, granularity);
    BCSRMatrix bcsr_probs(nullptr, tile_dense, N, N, granularity);
    
    softmax_bcsr_bcsr(bcsr_scores, bcsr_probs);

    // 3. Convert BCSR probs back to dense
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    time_and_print("bcsr_to_dense_kernel", [&]{
        BCSRView view_probs = bcsr_probs.get_view();
        bcsr_to_dense_kernel<<<grid, block>>>(view_probs, attn_probs, N, N);
    });
    cudaDeviceSynchronize();

    // 4. Dense attn_probs x V -> output
    matmul_tiled(attn_probs, V, output, N, N, d);
    cudaDeviceSynchronize();

    free(h_attn_scores);
    free(tile_dense);
    cudaFree(Q); cudaFree(K); cudaFree(V);
    cudaFree(k_transposed);
    cudaFree(attn_scores);
    cudaFree(attn_probs);
}
