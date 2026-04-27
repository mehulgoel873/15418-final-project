#include "matmul.cuh"
#include "datastructures/bcsr.cuh"
#include "timing.cuh"
#include <cassert>
#include <cstdio>
#include <cstring>

/// Naive matrix multiplication kernel: output = A x B
/// A: M x N, B: N x K, output: M x K
__global__ void matmul_kernel(float* A, float* B, float* output, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float value = 0.0f;
        for (int j = 0; j < N; j++) {
            value += A[row * N + j] * B[j * K + col];
        }
        output[row * K + col] = value;
    }
}


// SpMM: sparse BCSR A (M x N) * dense B (N x K) -> dense output (M x K).
// One block per scalar row of A, BN output columns per block. Each block walks
// its row's K_b * T scalar nonzeros in chunks of TK = 32, gathers the
// corresponding B-rows into shared memory, and accumulates into a per-thread
// register. Synchronous version (no cp.async) — see plan for the pipelined
// follow-up.
constexpr int SPMM_BN = 128;
constexpr int SPMM_TK = 32;

__global__ void spmm_bcsr_kernel(BCSR A, const float* __restrict__ B,
                                 float* __restrict__ output,
                                 int M, int K) {
    const int r        = blockIdx.x;                  // scalar row of A
    const int col_tile = blockIdx.y * SPMM_BN;        // first output col handled
    const int tid      = threadIdx.x;                 // also output col offset

    const int T   = A.TILING;
    const int bi  = r / T;
    const int ti  = r % T;
    const int K_b = A.block_row_K(bi);
    const bool active_col = (col_tile + tid) < K;

    if (K_b == 0) {
        if (active_col) output[(size_t)r * K + col_tile + tid] = 0.0f;
        return;
    }

    const int    nnz            = K_b * T;
    const size_t row_strip_base = A.block_row_base(bi) + (size_t)ti * K_b * T;
    const int    row_ptr_base   = A.row_ptr[bi];

    __shared__ float sA_val[SPMM_TK];
    __shared__ int   sA_col[SPMM_TK];
    __shared__ float sB[SPMM_TK][SPMM_BN + 4];   // +4 padding kills bank conflicts

    float acc = 0.0f;
    const int num_chunks = (nnz + SPMM_TK - 1) / SPMM_TK;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int p_start    = chunk * SPMM_TK;
        const int chunk_size = min(SPMM_TK, nnz - p_start);

        // A-load: first warp pulls 32 contiguous scalars from this row's strip.
        // Lanes that fall past the row's tail pad with zero so the FMA is a no-op.
        if (tid < SPMM_TK) {
            if (tid < chunk_size) {
                int p = p_start + tid;
                sA_val[tid] = A.values[row_strip_base + p];
                sA_col[tid] = A.col_idx[row_ptr_base + (p / T)] * T + (p % T);
            } else {
                sA_val[tid] = 0.0f;
                sA_col[tid] = 0;
            }
        }
        __syncthreads();

        // B-gather: each thread loads its column from each of the chunk's 32
        // B-rows. Coalesced across the warp; predicated for the K-tail.
        for (int t = 0; t < SPMM_TK; t++) {
            float v = 0.0f;
            if (t < chunk_size && active_col) {
                v = B[(size_t)sA_col[t] * K + col_tile + tid];
            }
            sB[t][tid] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < SPMM_TK; t++) {
            acc += sA_val[t] * sB[t][tid];
        }
        __syncthreads();   // before next iteration overwrites sA/sB
    }

    if (active_col) output[(size_t)r * K + col_tile + tid] = acc;
}

void spmm(BCSR& A, float* B, float* output, int M, int N, int K) {
    assert(A.M == M && A.N == N);
    cudaDeviceSynchronize();
    char label[64];
    snprintf(label, sizeof(label), "spmm %dx%dx%d", M, N, K);
    dim3 grid(M, (K + SPMM_BN - 1) / SPMM_BN);
    dim3 block(SPMM_BN);
    time_and_print(label, [&]{ spmm_bcsr_kernel<<<grid, block>>>(A, B, output, M, K); });
}

/// Host launcher for the naive matmul kernel.
void matmul_naive(float* A, float* B, float* output, int M, int N, int K) {
    char label[64];
    snprintf(label, sizeof(label), "matmul_naive %dx%dx%d", M, N, K);
    dim3 blockSize(16, 16);
    dim3 gridSize((K + 15) / 16, (M + 15) / 16);
    time_and_print(label, [&]{ matmul_kernel<<<gridSize, blockSize>>>(A, B, output, M, N, K); });
}


static constexpr int TILE_H      = 32;
static constexpr int TILE_W      = 32;
static constexpr int TILE_K      = 32;
static constexpr int NUM_BATCH   = 4;
static constexpr int TILE_K_STEP = TILE_K * NUM_BATCH;

__global__ void matmul_tiled_kernel(float* A, float* B, float* output, int M, int N, int K) {
    /* 
    VERSION 1
    __shared__ float sA[TILE_HEIGHT][TILE_K];   // [32][32]
    __shared__ float sB[TILE_K][TILE_WIDTH];    // [32][32]

    int row = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH  + threadIdx.x;

    float val = 0.0f;

    int numTiles = N / TILE_K;
    for (int t = 0; t < numTiles; t++) {
        // Each thread loads exactly 1 element from A and 1 from B — no logic needed.
        sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_K + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(t * TILE_K + threadIdx.y) * K + col];

        __syncthreads();

        for (int j = 0; j < TILE_K; j++) {
            val += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }

        __syncthreads();
    }

    output[row * K + col] = val;
    */

    // VERSION 2
    __shared__ float sA[TILE_H][TILE_K_STEP];
    __shared__ float sB[TILE_K_STEP][TILE_W];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE_H + ty;
    int col = blockIdx.x * TILE_W  + tx;

    float val = 0.0f;

    for (int t = 0; t < N / TILE_K_STEP; t++) {
        int k_base = t * TILE_K_STEP;
        for (int i = 0; i < NUM_BATCH; i++)
            sA[ty][tx + i * TILE_W] = A[row * N + k_base + tx + i * TILE_W];
        for (int i = 0; i < NUM_BATCH; i++)
            sB[ty + i * TILE_H][tx] = B[(k_base + ty + i * TILE_H) * K + col];
        __syncthreads();
        for (int j = 0; j < TILE_K_STEP; j++)
            val += sA[ty][j] * sB[j][tx];
        __syncthreads();
    }

    output[row * K + col] = val;
}

void matmul_tiled(float* A, float* B, float* output, int M, int N, int K) {
    if (M % TILE_H != 0 || K % TILE_W != 0 || N % TILE_K_STEP != 0) {
        fprintf(stderr,
                "matmul_tiled: dimensions must be divisible by tile size; got M=%d N=%d K=%d\n",
                M, N, K);
        assert(false);
    }

    char label[64];
    snprintf(label, sizeof(label), "matmul_tiled %dx%dx%d", M, N, K);
    dim3 blockSize(TILE_W, TILE_H);
    dim3 gridSize(K / TILE_W, M / TILE_H);
    time_and_print(label, [&]{ matmul_tiled_kernel<<<gridSize, blockSize>>>(A, B, output, M, N, K); });
}


__global__ void matmul_sparse_bcsr_kernel(BCSR A, BCSR B, BCSR output) {
    int bi = blockIdx.y, bj = blockIdx.x;
    int out_idx = output.block_idx[bi * output.num_block_cols + bj];
    if (out_idx < 0) return; // Doestn' exist on output sparse mask

    __shared__ float sA[TILE_H][TILE_K_STEP];
    __shared__ float sB[TILE_K_STEP][TILE_W];

    int ty = threadIdx.y, tx = threadIdx.x;
    float val = 0.0f;

    int row_start = A.row_ptr[bi], row_end = A.row_ptr[bi + 1];

    for (int k_base = row_start; k_base < row_end; k_base += NUM_BATCH) {
        for (int i = 0; i < NUM_BATCH; i++) {
            int k = k_base + i;
            if (k < row_end) {
                int bk = A.col_idx[k];
                int b_idx = B.block_idx[bk * B.num_block_cols + bj];
                sA[ty][tx + i * TILE_W] = A.values[k * TILE_H * TILE_W + ty * TILE_W + tx];
                sB[ty + i * TILE_H][tx] = (b_idx >= 0)
                    ? B.values[b_idx * TILE_H * TILE_W + ty * TILE_W + tx]
                    : 0.0f;
            } else {
                sA[ty][tx + i * TILE_W] = 0.0f;
                sB[ty + i * TILE_H][tx] = 0.0f;
            }
        }
        __syncthreads();
        for (int j = 0; j < TILE_K_STEP; j++)
            val += sA[ty][j] * sB[j][tx];
        __syncthreads();
    }

    output.values[out_idx * TILE_H * TILE_W + ty * TILE_W + tx] = val;
}

void matmul_sparse_bcsr(BCSR& A, BCSR& B, BCSR& output) {
    cudaDeviceSynchronize();
    char label[64];
    snprintf(label, sizeof(label), "matmul_sparse_bcsr %dx%dx%d", A.M, A.N, B.N);
    dim3 blockSize(TILE_W, TILE_H);
    dim3 gridSize(output.num_block_cols, output.num_block_rows);
    time_and_print(label, [&]{ matmul_sparse_bcsr_kernel<<<gridSize, blockSize>>>(A, B, output); });
}


// One thread per element: mat[i,j] *= mask[i,j].
__global__ void elemwise_mul_kernel(float* mat, const float* mask, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    int idx = row * n + col;
    mat[idx] *= mask[idx];
}


// In-place elementwise multiply of two n x n device matrices: mat *= mask.
// Used to apply a mask produced by get_rand_mask (1.0 keeps, -inf zeros-out
// after softmax when added to logits, or propagates -inf if multiplied here).
static void elementwise_mul(float* mat, float* mask, int n) {
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    elemwise_mul_kernel<<<grid, block>>>(mat, mask, n);
    cudaDeviceSynchronize();
}