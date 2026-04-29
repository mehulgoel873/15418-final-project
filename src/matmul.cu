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

/* 
VERSION 1
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
 */


// SpMM: sparse BCSR A (M x N) * dense B (N x K) -> dense output (M x K).
//
// One block per (block-row, output-column-tile). 2D thread layout:
// threadIdx.y = scalar row within the block-row, threadIdx.x = output column
// within the tile. All T scalar rows in a block-row share the same column-tile
// pattern (BCSR invariant), so the B-tile is loaded once per chunk and reused
// T times — T-fold reduction in B-bandwidth versus a per-scalar-row design.
//
// Each thread accumulates exactly one fp32 output. No cp.async, no
// double-buffering; the warp scheduler hides DRAM latency by overlapping
// load-warps with compute-warps within a block.
//
// Block sizing scales with T:
//   T <= 2 → 256 threads (otherwise BN_COLS would blow sB past shared-mem cap)
//   T >= 4 → 1024 threads (more rows means we can spend the bigger thread
//                          budget on wider, fully-coalesced output tiles)
constexpr int SPMM_TK = 32;

template<int T, int BN_COLS>
__global__ void spmm_bcsr_kernel(BCSRView A, const float* __restrict__ B,
                                 float* __restrict__ output,
                                 int M, int K) {
    const int bi       = blockIdx.x;                  // block-row of A
    const int col_tile = blockIdx.y * BN_COLS;        // first output col
    const int ti       = threadIdx.y;                 // row within block-row
    const int c        = threadIdx.x;                 // col within tile
    const int tid      = ti * BN_COLS + c;
    constexpr int NTHREADS = T * BN_COLS;

    const int K_b = A.block_row_K(bi);
    const int r   = bi * T + ti;
    const int col = col_tile + c;
    const bool active_out = (r < M) && (col < K);

    if (K_b == 0) {
        if (active_out) output[(size_t)r * K + col] = 0.0f;
        return;
    }

    const int    nnz          = K_b * T;
    const size_t br_base      = A.block_row_base(bi);
    const int    row_ptr_base = A.row_ptr[bi];

    __shared__ float sA_val[T][SPMM_TK];
    __shared__ int   sA_col[SPMM_TK];
    __shared__ float sB    [SPMM_TK][BN_COLS];

    float acc = 0.0f;
    const int num_chunks = (nnz + SPMM_TK - 1) / SPMM_TK;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int p_start    = chunk * SPMM_TK;
        const int chunk_size = min(SPMM_TK, nnz - p_start);

        // sA_col: shared across all T rows. Only TK entries — first warp.
        if (tid < SPMM_TK) {
            if (tid < chunk_size) {
                int p = p_start + tid;
                sA_col[tid] = A.col_idx[row_ptr_base + (p / T)] * T + (p % T);
            } else {
                sA_col[tid] = 0;
            }
        }

        // sA_val: T*TK floats, distributed across all NTHREADS threads.
        for (int idx = tid; idx < T * SPMM_TK; idx += NTHREADS) {
            int t_i = idx / SPMM_TK;
            int j   = idx % SPMM_TK;
            int p   = p_start + j;
            sA_val[t_i][j] = (j < chunk_size)
                ? A.values[br_base + (size_t)t_i * K_b * T + p]
                : 0.0f;
        }
        __syncthreads();   // sA_col must be visible before B load uses it

        // sB: TK*BN_COLS floats, distributed across NTHREADS threads.
        for (int idx = tid; idx < SPMM_TK * BN_COLS; idx += NTHREADS) {
            int t_i = idx / BN_COLS;
            int cc  = idx % BN_COLS;
            int gcol = col_tile + cc;
            if (t_i < chunk_size && gcol < K) {
                sB[t_i][cc] = B[(size_t)sA_col[t_i] * K + gcol];
            } else {
                sB[t_i][cc] = 0.0f;
            }
        }
        __syncthreads();   // sB visible before compute

        #pragma unroll
        for (int t_i = 0; t_i < SPMM_TK; t_i++) {
            acc += sA_val[ti][t_i] * sB[t_i][c];
        }
        __syncthreads();   // before next chunk overwrites shared mem
    }

    if (active_out) output[(size_t)r * K + col] = acc;
}

void spmm(BCSRMatrix& A, float* B, float* output, int M, int N, int K) {
    assert(A.get_M() == M && A.get_N() == N);
    cudaDeviceSynchronize();
    
    BCSRView viewA = A.get_view();
    
    char label[64];
    snprintf(label, sizeof(label), "spmm %dx%dx%d (T=%d)", M, N, K, viewA.TILING);
    const int T = viewA.TILING;

    auto launch = [&](auto kernel, int bn_cols, int T_rt) {
        dim3 grid(M / T_rt, (K + bn_cols - 1) / bn_cols);
        dim3 block(bn_cols, T_rt);
        time_and_print(label, [&]{ kernel<<<grid, block>>>(viewA, B, output, M, K); });
    };

    switch (T) {
        case 1:  launch(spmm_bcsr_kernel<1,  256>, 256, 1);  break;
        case 2:  launch(spmm_bcsr_kernel<2,  128>, 128, 2);  break;
        case 4:  launch(spmm_bcsr_kernel<4,  256>, 256, 4);  break;
        case 8:  launch(spmm_bcsr_kernel<8,  128>, 128, 8);  break;
        case 16: launch(spmm_bcsr_kernel<16,  64>,  64, 16); break;
        case 32: launch(spmm_bcsr_kernel<32,  32>,  32, 32); break;
        default:
            fprintf(stderr, "spmm: unsupported TILING=%d (must be 1,2,4,8,16,32)\n", T);
            assert(false);
    }
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