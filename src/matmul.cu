#include "matmul.cuh"
#include "datastructures/bcsr.cuh"
#include "timing.cuh"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda_pipeline_primitives.h>

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
// register.
//
// Pipelined: A-chunk and B-tile are both double-buffered. While compute runs
// on the current chunk's data in buffer `buf`, the next chunk's data is in
// flight via cp.async into buffer `1-buf`. B-gather uses 16-byte cp.async
// (the only width that actually skips the register file on Ampere+); 4 warps
// each handle 8 chunk-rows, with each warp's 32 lanes loading one row of 128
// floats as float4 per lane.
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

    __shared__ float sA_val[2][SPMM_TK];
    __shared__ int   sA_col[2][SPMM_TK];
    __shared__ float sB[2][SPMM_TK][SPMM_BN + 4];   // +4 padding kills bank conflicts

    // Synchronous: pull TK scalars from the row's strip into sA_val/sA_col[buf].
    // Only the first warp does work; others wait at the __syncthreads.
    auto load_A_chunk = [&](int chunk_idx, int buf) {
        const int p_start    = chunk_idx * SPMM_TK;
        const int chunk_size = min(SPMM_TK, nnz - p_start);
        if (tid < SPMM_TK) {
            if (tid < chunk_size) {
                int p = p_start + tid;
                sA_val[buf][tid] = A.values[row_strip_base + p];
                sA_col[buf][tid] = A.col_idx[row_ptr_base + (p / T)] * T + (p % T);
            } else {
                sA_val[buf][tid] = 0.0f;
                sA_col[buf][tid] = 0;
            }
        }
        __syncthreads();   // sA_col[buf] must be visible before issue_B_load reads it
    };

    // Async: issue cp.async loads for the chunk's 32 B-rows into sB[buf].
    // Layout: 4 warps × 8 chunk-rows each; each warp's 32 lanes load one row
    // of 128 floats as float4. Tail predication for K not divisible by BN
    // falls back to per-element scalar copies.
    auto issue_B_load = [&](int chunk_idx, int buf) {
        const int p_start    = chunk_idx * SPMM_TK;
        const int chunk_size = min(SPMM_TK, nnz - p_start);
        const int warp_id    = tid / 32;
        const int lane       = tid % 32;
        const int col_base   = col_tile + lane * 4;

        for (int t = warp_id; t < SPMM_TK; t += 4) {
            if (t >= chunk_size) {
                *reinterpret_cast<float4*>(&sB[buf][t][lane * 4]) = make_float4(0, 0, 0, 0);
                continue;
            }
            const size_t b_row_off = (size_t)sA_col[buf][t] * K;
            if (col_base + 3 < K) {
                __pipeline_memcpy_async(
                    &sB[buf][t][lane * 4],
                    &B[b_row_off + col_base],
                    16);
            } else {
                // K-tail: load up to 4 floats with bounds, pad rest with zero.
                float4 v = make_float4(0, 0, 0, 0);
                float* vp = reinterpret_cast<float*>(&v);
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    if (col_base + e < K) vp[e] = B[b_row_off + col_base + e];
                }
                *reinterpret_cast<float4*>(&sB[buf][t][lane * 4]) = v;
            }
        }
    };

    float acc = 0.0f;
    const int num_chunks = (nnz + SPMM_TK - 1) / SPMM_TK;
    int buf = 0;

    // Prologue: prime buffer 0 with chunk 0's data.
    load_A_chunk(0, 0);
    issue_B_load(0, 0);
    __pipeline_commit();

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int next_buf = 1 - buf;

        // Wait for the most recently committed cp.async group (current chunk's B).
        __pipeline_wait_prior(0);
        __syncthreads();

        // Issue next chunk's loads while compute runs on the current chunk.
        if (chunk + 1 < num_chunks) {
            load_A_chunk(chunk + 1, next_buf);
            issue_B_load(chunk + 1, next_buf);
            __pipeline_commit();
        }

        #pragma unroll
        for (int t = 0; t < SPMM_TK; t++) {
            acc += sA_val[buf][t] * sB[buf][t][tid];
        }

        buf = next_buf;
    }

    if (active_col) output[(size_t)r * K + col_tile + tid] = acc;
}

void spmm(BCSR& A, float* B, float* output, int M, int N, int K) {
    assert(A.M == M && A.N == N);
    assert(K % 4 == 0);   // 16-byte cp.async requires 4-float alignment on B rows
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