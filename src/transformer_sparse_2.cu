#include "transformer_sparse_2.cuh"
#include "transformer_naive.cuh"
#include "matmul.cuh"
#include "softmax.cuh"
#include "timing.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>

static void rand_init_device_buf(float* d_ptr, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX;
    cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

__global__ static void build_tile_mask_kernel_v2(const float* mask, bool* tile_dense, int N, int granularity, int Tb) {
    int bj = blockIdx.x * blockDim.x + threadIdx.x;
    int bi = blockIdx.y * blockDim.y + threadIdx.y;

    if (bi < Tb && bj < Tb) {
        bool any_finite = false;
        for (int ti = 0; ti < granularity && !any_finite; ti++) {
            for (int tj = 0; tj < granularity && !any_finite; tj++) {
                float v = mask[(bi * granularity + ti) * N + (bj * granularity + tj)];
                if (isfinite(v)) any_finite = true;
            }
        }
        tile_dense[bi * Tb + bj] = any_finite;
    }
}

static bool* build_tile_mask_from_additive_v2(const float* d_mask, int N, int granularity) {
    int Tb = N / granularity;
    bool* d_tile_dense;
    cudaMalloc(&d_tile_dense, Tb * Tb * sizeof(bool));
    dim3 block(16, 16);
    dim3 grid((Tb + 15) / 16, (Tb + 15) / 16);
    build_tile_mask_kernel_v2<<<grid, block>>>(d_mask, d_tile_dense, N, granularity, Tb);
    return d_tile_dense;
}

TransformerSparse2::TransformerSparse2(int d) : d_W_q(nullptr), d_W_k(nullptr), d_W_v(nullptr), d_dim(d) {
    size_t bytes = (size_t)d * d * sizeof(float);
    cudaMalloc(&d_W_q, bytes);
    cudaMalloc(&d_W_k, bytes);
    cudaMalloc(&d_W_v, bytes);
    rand_init_device_buf(d_W_q, d * d);
    rand_init_device_buf(d_W_k, d * d);
    rand_init_device_buf(d_W_v, d * d);
}

TransformerSparse2::~TransformerSparse2() {
    cudaFree(d_W_q);
    cudaFree(d_W_k);
    cudaFree(d_W_v);
}

void TransformerSparse2::forward(float* x, float* mask, float* output, int N, int d, int granularity) {
    float *Q, *K, *V;
    size_t tok_bytes = (size_t)N * d * sizeof(float);
    if (cudaMalloc(&Q, tok_bytes) != cudaSuccess ||
        cudaMalloc(&K, tok_bytes) != cudaSuccess ||
        cudaMalloc(&V, tok_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: N=%d d=%d\n", N, d);
        return;
    }

    matmul_tiled(x, d_W_q, Q, N, d, d);
    matmul_tiled(x, d_W_k, K, N, d, d);
    matmul_tiled(x, d_W_v, V, N, d, d);

    // Gather-based SDDMM (sddmm.cu): treats B's rows as B^T's columns, so K
    // does NOT need to be transposed. Only sampled tiles are computed.
    bool* d_tile_dense = build_tile_mask_from_additive_v2(mask, N, granularity);
    BCSRMatrix scores_bcsr(nullptr, d_tile_dense, N, N, granularity);
    BCSRMatrix probs_bcsr (nullptr, d_tile_dense, N, N, granularity);

    sddmm(Q, K, scores_bcsr, d);                                  // 4-arg form -> sddmm.cu
    scale_bcsr_values(scores_bcsr, 1.0f / sqrtf((float)d));
    softmax_bcsr_bcsr(scores_bcsr, probs_bcsr);

    spmm(probs_bcsr, V, output, N, N, d);

    cudaFree(Q); cudaFree(K); cudaFree(V);
    cudaFree(d_tile_dense);
}
