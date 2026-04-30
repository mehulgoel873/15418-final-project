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

__global__ void build_tile_mask_kernel(const float* mask, bool* tile_dense, int N, int granularity, int Tb) {
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

// Build a granularity-block sparsity mask from the additive softmax mask:
// a tile is dense iff any entry inside it is finite (i.e. not all -INF).
static bool* build_tile_mask_from_additive(const float* d_mask, int N, int granularity) {
    int Tb = N / granularity;
    
    bool* d_tile_dense;
    cudaMalloc(&d_tile_dense, Tb * Tb * sizeof(bool));
    
    dim3 block(16, 16);
    dim3 grid((Tb + 15) / 16, (Tb + 15) / 16);
    build_tile_mask_kernel<<<grid, block>>>(d_mask, d_tile_dense, N, granularity, Tb);
    
    return d_tile_dense;
}

__global__ void scale_Q_kernel(float* Q, int len, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        Q[idx] *= scale;
    }
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
    float *Q, *K, *V, *K_T;
    size_t tok_bytes  = (size_t)N * d * sizeof(float);
    if (cudaMalloc(&Q,      tok_bytes)  != cudaSuccess ||
        cudaMalloc(&K,      tok_bytes)  != cudaSuccess ||
        cudaMalloc(&V,      tok_bytes)  != cudaSuccess ||
        cudaMalloc(&K_T,    tok_bytes)  != cudaSuccess) {
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

    // Scale Q by 1/sqrt(d) instead of scaling the full NxN attention matrix later
    // TODO: fuse this operation into the softmax
    int num_elements = N * d;
    scale_Q_kernel<<<(num_elements + 255) / 256, 256>>>(Q, num_elements, 1.0f / sqrtf((float)d));
    cudaDeviceSynchronize();

    // Pack dense probs into BCSR using a tile-mask derived from the additive
    // mask: tiles that are entirely -INF in `mask` are dropped, since their
    // softmax outputs are zero and contribute nothing to probs @ V.
    bool* d_tile_dense = build_tile_mask_from_additive(mask, N, granularity);
    
    BCSRMatrix scores_bcsr(nullptr, d_tile_dense, N, N, granularity);
    BCSRMatrix probs_bcsr(nullptr, d_tile_dense, N, N, granularity);

    // scores = Q K^T : (N x d) dense @ (d x N) dense -> (N x N) sparse
    sddmm(Q, K_T, scores_bcsr, N, d, N);

    // probs = softmax(scores) row-wise keeping the same sparsity pattern
    softmax_bcsr_bcsr(scores_bcsr, probs_bcsr);

    // attn_out = probs_bcsr @ V : (N x N) sparse @ (N x d) dense -> (N x d) dense
    spmm(probs_bcsr, V, output, N, N, d);

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(K_T);
    cudaFree(d_tile_dense);
}
