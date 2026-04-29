#include "transformer_sparse.cuh"
#include "transformer_naive.cuh"
#include "matmul.cuh"
#include "softmax.cuh"
#include "timing.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>

static void rand_init_device_buf(float* d_ptr, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX;
    cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

// Build a granularity-block sparsity mask from the additive softmax mask:
// a tile is dense iff any entry inside it is finite (i.e. not all -INF).
static bool* build_tile_mask_from_additive(const float* d_mask, int N, int granularity) {
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

TransformerSparse::TransformerSparse(int d) : d_W_q(nullptr), d_W_k(nullptr), d_W_v(nullptr), d_dim(d) {
    size_t bytes = (size_t)d * d * sizeof(float);
    cudaMalloc(&d_W_q, bytes);
    cudaMalloc(&d_W_k, bytes);
    cudaMalloc(&d_W_v, bytes);
    rand_init_device_buf(d_W_q, d * d);
    rand_init_device_buf(d_W_k, d * d);
    rand_init_device_buf(d_W_v, d * d);
}

TransformerSparse::~TransformerSparse() {
    cudaFree(d_W_q);
    cudaFree(d_W_k);
    cudaFree(d_W_v);
}

void TransformerSparse::forward(float* x, float* mask, float* output, int N, int d, int granularity) {
    float *Q, *K, *V, *K_T, *scores, *probs;
    size_t tok_bytes  = (size_t)N * d * sizeof(float);
    size_t attn_bytes = (size_t)N * N * sizeof(float);
    if (cudaMalloc(&Q,      tok_bytes)  != cudaSuccess ||
        cudaMalloc(&K,      tok_bytes)  != cudaSuccess ||
        cudaMalloc(&V,      tok_bytes)  != cudaSuccess ||
        cudaMalloc(&K_T,    tok_bytes)  != cudaSuccess ||
        cudaMalloc(&scores, attn_bytes) != cudaSuccess ||
        cudaMalloc(&probs,  attn_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: N=%d d=%d\n", N, d);
        return;
    }

    // Project x into Q, K, V: (N x d) @ (d x d) -> (N x d).
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

    // Pack dense probs into BCSR using a tile-mask derived from the additive
    // mask: tiles that are entirely -INF in `mask` are dropped, since their
    // softmax outputs are zero and contribute nothing to probs @ V.
    bool* tile_dense = build_tile_mask_from_additive(mask, N, granularity);
    float* h_probs = (float*)malloc(attn_bytes);
    cudaMemcpy(h_probs, probs, attn_bytes, cudaMemcpyDeviceToHost);
    BCSRMatrix probs_bcsr(h_probs, tile_dense, N, N, granularity);
    free(h_probs);
    free(tile_dense);

    // attn_out = probs_bcsr @ V : (N x N) sparse @ (N x d) dense -> (N x d) dense
    spmm(probs_bcsr, V, output, N, N, d);

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(K_T);
    cudaFree(scores); cudaFree(probs);
}
