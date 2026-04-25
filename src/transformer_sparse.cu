#include "transformer_sparse.cuh"
#include "transformer_naive.cuh"
#include "matmul.cuh"
#include "softmax.cuh"
#include "timing.cuh"
#include <stdio.h>

void TransformerSparse::forward(float* q, float* k, float* v, float* output, int N, int d, const bool* mask, int granularity) {
    float* k_transposed;
    size_t tok_bytes  = (size_t)N * d * sizeof(float);
    if (cudaMalloc(&k_transposed, tok_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for k_transposed\n");
        return;
    }

    dim3 blockSize16(16, 16);
    dim3 gridSize16((N + 15) / 16, (N + 15) / 16);
    transpose_kernel<<<gridSize16, blockSize16>>>(k, k_transposed, N, d);
    cudaDeviceSynchronize();

    // Create BCSR topologies using the provided mask
    BCSR attn_scores(nullptr, mask, N, N, granularity);
    BCSR attn_probs(nullptr, mask, N, N, granularity);

    // 1. SDDMM: Q x K^T -> attn_scores
    // !!PLACEHOLDER!! SDDMM: A (M x d) times B (d x N) -> Sparse Output (M x N)

    // 2. Sparse Softmax
    softmax_bcsr_bcsr(attn_scores, attn_probs);

    // 3. SpMM: attn_probs x V -> output
    // !!PLACEHOLDER!! SpMM: Sparse A (M x N) times Dense B (N x d) -> Dense Output (M x d)
    matmul_sparse(attn_probs, v, output, N, d, N);

    cudaFree(k_transposed);
}
