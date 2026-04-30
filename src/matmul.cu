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


// SDDMM: dense A (M x N) * dense B (N x K) -> sparse BCSR output (M x K).
// Each block has THREADS_PER_BLOCK threads. THREADS_PER_BLOCK / 32 tiles are assigned to each block.
// Each warp handles one TxT tile of the output. 
//
// The kernel iterates over the N dimension in chunks of TILE_W, 
// loading the corresponding tiles of A and B into shared memory, 
// computing partial products for indices corresponing to nonzero positions in C's BCSR structure,
// and accumulating into shared memory.
// After processing all N-chunks, each thread writes its result to the appropriate location in C.
//
// K must be divisible by MAX_TILE_DIM for correct indexing.
//
// Shared memory usage per block:
// - sA: (THREADS_PER_BLOCK / 32) * MAX_TILE_DIM * MAX_TILE_DIM floats
// - sB: (THREADS_PER_BLOCK / 32) * MAX_TILE_DIM * MAX_TILE_DIM floats
// - sC: (THREADS_PER_BLOCK / 32) * MAX_TILE_DIM * MAX_TILE_DIM floats
// Total: 3 * (THREADS_PER_BLOCK / 32) * MAX_TILE_DIM^2 * 4 bytes = 98304 bytes

/* VERSION 3: use active coordinate list to skip zero tiles, better than version 2 on lower granularities, can be improved with smaller tile sizes due to shared memory limits but incompatible with granularity 32 */
template<int TILE_H, int TILE_W, int WARPS_PER_BLOCK>
__global__ void sddmm_bcsr_kernel(float* A, float* B, BCSRView C, int M, int N, int K, int* tile_counter) {    
    extern __shared__ float shared_mem[];

    typedef float (*FloatArray)[TILE_H][TILE_W];
    typedef int (*Int2DArray)[TILE_H * TILE_W];
    typedef int (*Int1DArray);

    // Shared memory for the A and B tiles, and the partial products for C
    FloatArray sA = (FloatArray)shared_mem;
    FloatArray sB = (FloatArray)&shared_mem[1 * WARPS_PER_BLOCK * TILE_H * TILE_W];
    FloatArray sC = (FloatArray)&shared_mem[2 * WARPS_PER_BLOCK * TILE_H * TILE_W]; // for accumulating partial products corresponding to nonzero positions in C

    // Shared memory for the Active Coordinate List
    Int2DArray s_active_idx = (Int2DArray)(int*)&shared_mem[3 * WARPS_PER_BLOCK * TILE_H * TILE_W];
    Int1DArray s_num_active = (Int1DArray)(int*)&shared_mem[4 * WARPS_PER_BLOCK * TILE_H * TILE_W]; // one int per warp for counting active coordinates

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int T_c = C.TILING; // <= 32
    
    int num_tile_cols = (K + TILE_W - 1) / TILE_W;
    int num_total_tiles = ((M + TILE_H - 1) / TILE_H) * num_tile_cols;

    while (true) {
        // 1. Dynamic SDDMM Tile Scheduling
        int tile_idx;
        if (lane_id == 0) {
            tile_idx = atomicAdd(tile_counter, 1);
        }
        tile_idx = __shfl_sync(0xFFFFFFFF, tile_idx, 0);

        if (tile_idx >= num_total_tiles) break; // all tiles processed

        int tile_row = tile_idx / num_tile_cols;
        int tile_col = tile_idx % num_tile_cols;

        // Map the 32x32 SDDMM tile bounds to BCSR block bounds
        int bi_start = (tile_row * TILE_H) / T_c;
        int bi_end = min(((tile_row + 1) * TILE_H + T_c - 1) / T_c, C.num_block_rows);
        int bj_start = (tile_col * TILE_W) / T_c;
        int bj_end = ((tile_col + 1) * TILE_W + T_c - 1) / T_c;

        // 2. Sparse Rejection, Building Active Coordinate List
        if (lane_id == 0) s_num_active[warp_id] = 0;
        __syncwarp();

        // Iterate over active blocks and append them to the active coordinate list
        for (int bi = bi_start; bi < bi_end; bi++) {
            for (int bj = bj_start; bj < bj_end; bj++) {
                int j_local = C.rev_col_idx[bi * C.num_block_cols + bj];
                if (j_local >= 0) {
                    // Warp extracts coordinates of this T_c x T_c block
                    for (int i = lane_id; i < T_c * T_c; i += 32) {
                        int r_local = i / T_c;
                        int c_local = i % T_c;
                        
                        int sC_r = (bi * T_c - tile_row * TILE_H) + r_local;
                        int sC_c = (bj * T_c - tile_col * TILE_W) + c_local;
                        
                        // Flatten 2D coordinate to 1D index
                        int flat_idx = sC_r * TILE_W + sC_c;

                        // Atomically append to the list
                        int base_current_iter = s_num_active[warp_id] + (i / 32) * 32;
                        // s_num_active[warp_id] is updated once after the loop
                        s_active_idx[warp_id][base_current_iter + lane_id] = flat_idx;

                    
                        /* [BAD] Version 2: slower on granularity 32, breaks on lower granularities
                        unsigned mask = __ballot_sync(0xFFFFFFFF, 1);
                        int my_rank = __popc(mask << (32 - lane_id)) ; // # active lanes before current
                        // equivalent to: popcount of bits [0, lane_id)
                        int prefix = __popc(mask & ((1u << lane_id) - 1));
                        int base;
                        if (lane_id == 0) {
                            base = s_num_active[warp_id];
                            s_num_active[warp_id] += __popc(mask); // atomic-free write
                        }
                        base = __shfl_sync(0xFFFFFFFF, base, 0);

                        s_active_idx[warp_id][base + prefix] = flat_idx; 
                        */

                        /* Version 1: atomic append with higher contention
                        int append_idx = atomicAdd(&s_num_active[warp_id], 1);
                        s_active_idx[warp_id][append_idx] = flat_idx; 
                        */
                    }
                    if (lane_id == 0) s_num_active[warp_id] += T_c * T_c;  // one write at end of block
                    __syncwarp();
                }
            }
        }
        __syncwarp();

        // If the 32x32 tile has zero active elements, instantly skip to the next tile
        int active_count = s_num_active[warp_id];
        if (active_count == 0) continue;

        // 3. Dense Load, Sparse Compute
        // Clear the active elements in sC
        for (int i = lane_id; i < active_count; i += 32) {
            int flat_idx = s_active_idx[warp_id][i];
            sC[warp_id][flat_idx / TILE_W][flat_idx % TILE_W] = 0.0f;
        }
        __syncwarp();

        // SDDMM Loop
        for (int t = 0; t < (N + TILE_W - 1) / TILE_W; t++) {
            
            // Dense Load
            for (int i = lane_id; i < TILE_H * TILE_W; i += 32) {
                int r = i / TILE_W;
                int c = i % TILE_W;
                int g_row_A = tile_row * TILE_H + r;
                int g_col_A = t * TILE_W + c;
                int g_row_B = t * TILE_W + r;
                int g_col_B = tile_col * TILE_W + c;

                sA[warp_id][r][c] = (g_row_A < M && g_col_A < N) ? A[g_row_A * N + g_col_A] : 0.0f;
                sB[warp_id][r][c] = (g_row_B < N && g_col_B < K) ? B[g_row_B * K + g_col_B] : 0.0f;
            }
            __syncwarp();

            // Sparse Compute
            for (int i = lane_id; i < active_count; i += 32) {
                int flat_idx = s_active_idx[warp_id][i];
                int r = flat_idx / TILE_W;
                int c = flat_idx % TILE_W;

                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < TILE_W; k++) {
                    sum += sA[warp_id][r][k] * sB[warp_id][k][c];
                }
                sC[warp_id][r][c] += sum;
            }
            __syncwarp();
        }

\       // 4. Sparse Write-Back
        for (int bi = bi_start; bi < bi_end; bi++) {
            int row_start = C.row_ptr[bi];
            int num_blocks_in_row = C.row_ptr[bi + 1] - row_start;
            size_t base_offset = (size_t)row_start * T_c * T_c;

            for (int bj = bj_start; bj < bj_end; bj++) {
                int j_local = C.rev_col_idx[bi * C.num_block_cols + bj];
                if (j_local >= 0) {
                    for (int i = lane_id; i < T_c * T_c; i += 32) {
                        int r_local = i / T_c;
                        int c_local = i % T_c;
                        
                        int sC_r = (bi * T_c - tile_row * TILE_H) + r_local;
                        int sC_c = (bj * T_c - tile_col * TILE_W) + c_local;
                        
                        int idx = base_offset + r_local * num_blocks_in_row * T_c + j_local * T_c + c_local;
                                     
                        C.values[idx] = sC[warp_id][sC_r][sC_c];
                    }
                }
            }
        }
        __syncwarp();
    }
}

/* VERSION 2: one warp per sparse block, straightforward block mapping, performs poorly on low granularities
static constexpr int MAX_TILE_DIM = 32; 
__global__ void sddmm_bcsr_direct_kernel(float* A, float* B, BCSRView C, int M, int N, int K) {
    extern __shared__ float shared_mem[];

    typedef float (*FloatArray)[MAX_TILE_DIM][MAX_TILE_DIM];

    FloatArray sA = (FloatArray)shared_mem;
    FloatArray sB = (FloatArray)&shared_mem[1 * WARPS_PER_BLOCK_sddmm * MAX_TILE_DIM * MAX_TILE_DIM];
    FloatArray sC = (FloatArray)&shared_mem[2 * WARPS_PER_BLOCK_sddmm * MAX_TILE_DIM * MAX_TILE_DIM]; // for accumulating partial products corresponding to nonzero positions in C

    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int T = C.TILING; // <= 32

    if (global_warp_id >= C.nnzb) return;
    int blk_idx = global_warp_id;

    // Find BCSR Row (bi) using Binary Search
    int bi = 0;
    if (lane_id == 0) {
        int low = 0, high = C.num_block_rows - 1;
        while (low < high) {
            int mid = low + (high - low + 1) / 2;
            if (C.row_ptr[mid] <= blk_idx) {
                low = mid; 
            } else {
                high = mid - 1;
            }
        }
        bi = low;
    }
    bi = __shfl_sync(0xFFFFFFFF, bi, 0);

    // Resolve global starting coordinates
    int bj = C.col_idx[blk_idx];
    int tile_row_start = bi * T;
    int tile_col_start = bj * T;

    // Initialize accumulator for this T x T tile
    for (int i = lane_id; i < T * T; i += 32) {
        sC[warp_id][i / T][i % T] = 0.0f;
    }
    __syncwarp();

    // SDDMM Loop
    // uses only the top-left T x T corner of the 32x32 shared memory
    for (int t = 0; t < (N + T - 1) / T; t++) {
        
        // Load T x T tiles of A and B
        for (int i = lane_id; i < T * T; i += 32) {
            int r = i / T;
            int c = i % T;
            
            int g_row_A = tile_row_start + r;
            int g_col_A = t * T + c;
            int g_row_B = t * T + r;
            int g_col_B = tile_col_start + c;

            // TODO: remove bounds check because N,K divisible by T
            sA[warp_id][r][c] = (g_row_A < M && g_col_A < N) ? A[g_row_A * N + g_col_A] : 0.0f;
            sB[warp_id][r][c] = (g_row_B < N && g_col_B < K) ? B[g_row_B * K + g_col_B] : 0.0f;
        }
        __syncwarp(); 

        // Compute dense partial products within non-zero block
        for (int i = lane_id; i < T * T; i += 32) {
            int r = i / T;
            int c = i % T;
            float sum = 0.0f;
            for (int k = 0; k < T; k++) {
                sum += sA[warp_id][r][k] * sB[warp_id][k][c];
            }
            sC[warp_id][r][c] += sum;
        }
        __syncwarp(); 
    }

    // Write results back to C.values
    int row_start = C.row_ptr[bi];
    int num_blocks_in_row = C.row_ptr[bi + 1] - row_start;
    int j_local = blk_idx - row_start; 
    
    size_t base_offset = (size_t)row_start * T * T;

    for (int i = lane_id; i < T * T; i += 32) {
        int r = i / T;
        int c = i % T;
        
        if (tile_row_start + r < M && tile_col_start + c < K) {
            int idx = base_offset + r * num_blocks_in_row * T + j_local * T + c;
            C.values[idx] = sC[warp_id][r][c];
        }
    }
}
*/

/* VERSION 1: basically tiled matmul with a sparse output mask, fails on lower granularities
static constexpr int TILE_H_sddmm = 32;
static constexpr int TILE_W_sddmm = 32;
static constexpr int NUM_BLOCKS_sddmm = 2048;   // must be >= (M/TILE_H) * (K/TILE_W) * sparsity / 32 for full occupancy
static constexpr int THREADS_PER_BLOCK_sddmm = 256;
static constexpr int WARPS_PER_BLOCK_sddmm = THREADS_PER_BLOCK_sddmm / 32;

__global__ void sddmm_bcsr_kernel(float* A, float* B, BCSRView C, int M, int N, int K) {
    extern __shared__ float shared_mem[];

    typedef float (*FloatArray)[TILE_H_sddmm][TILE_W_sddmm];

    FloatArray sA = (FloatArray)shared_mem;
    FloatArray sB = (FloatArray)&shared_mem[1 * WARPS_PER_BLOCK_sddmm * TILE_H_sddmm * TILE_W_sddmm];
    FloatArray sC = (FloatArray)&shared_mem[2 * WARPS_PER_BLOCK_sddmm * TILE_H_sddmm * TILE_W_sddmm]; // for accumulating partial products corresponding to nonzero positions in C

    int warp_id = threadIdx.x / 32; 
    int lane_id = threadIdx.x % 32;

    // SDDMM mapping: each warp computes one tile of the output C
    // int tile_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    int total_tiles = ((M + TILE_H_sddmm - 1) / TILE_H_sddmm) * ((K + TILE_W_sddmm - 1) / TILE_W_sddmm);

    // wrap warps around to process the entire matrix
    for (int tile_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32; 
        tile_idx < total_tiles; 
        tile_idx += (gridDim.x * blockDim.x) / 32) {

        int tile_row = tile_idx / (K / TILE_W_sddmm);
        int tile_col = tile_idx % (K / TILE_W_sddmm);

        // Check if this tile is entirely out of bounds
        if (tile_row * TILE_H_sddmm >= M || tile_col * TILE_W_sddmm >= K) return;

        // Initialize the accumulator for this warp
        for (int i = lane_id; i < TILE_H_sddmm * TILE_W_sddmm; i += 32) {
            sC[warp_id][i / TILE_W_sddmm][i % TILE_W_sddmm] = 0.0f;
        }
        __syncwarp();

        for (int t = 0; t < (N + TILE_W_sddmm - 1) / TILE_W_sddmm; t++) {
            // Load tile of A and B into shared memory
            for (int i = lane_id; i < TILE_H_sddmm * TILE_W_sddmm; i += 32) {
                int local_row = i / TILE_W_sddmm;
                int local_col = i % TILE_W_sddmm;
                
                int global_row_A = tile_row * TILE_H_sddmm + local_row;
                int global_col_A = t * TILE_W_sddmm + local_col;
                
                int global_row_B = t * TILE_W_sddmm + local_row;
                int global_col_B = tile_col * TILE_W_sddmm + local_col;

                if (global_row_A < M && global_col_A < N) {
                    sA[warp_id][local_row][local_col] = A[global_row_A * N + global_col_A];
                } else {
                    sA[warp_id][local_row][local_col] = 0.0f;
                }

                if (global_row_B < N && global_col_B < K) {
                    sB[warp_id][local_row][local_col] = B[global_row_B * K + global_col_B];
                } else {
                    sB[warp_id][local_row][local_col] = 0.0f;
                }
            }
            __syncwarp();

            // Compute partial products for the tile
            // int nnz = C.nnzb * C.TILING * C.TILING;  // number of nonzeros in the tile
            // for (int i = threadIdx.x % 32; i < nnz; i += 32) {
            //     int out_b_idx = C.block_idx

            // }
            // Each thread calculates 8 elements (256 elements / 32 threads)
            for (int i = lane_id; i < TILE_H_sddmm * TILE_W_sddmm; i += 32) {
                int local_r = i / TILE_W_sddmm;
                int local_c = i % TILE_W_sddmm;
                
                float sum = 0.0f;
                // Dot product of sA's row and sB's column
                for (int k = 0; k < TILE_W_sddmm; k++) {
                    sum += sA[warp_id][local_r][k] * sB[warp_id][k][local_c];
                }
                sC[warp_id][local_r][local_c] += sum;
            }
            __syncwarp();
        }

        // Write the result to the respective location in C if it's a nonzero position
        // Each warp handles writes to a TxT tile of the output
        for (int i = lane_id; i < TILE_H_sddmm * TILE_W_sddmm; i += 32) {
            int local_r = i / TILE_W_sddmm;
            int local_c = i % TILE_W_sddmm;
            
            int global_row = tile_row * TILE_H_sddmm + local_r;
            int global_col = tile_col * TILE_W_sddmm + local_c;
            
            if (global_row < M && global_col < K) {
                // Find the BCSR block coordinates
                int bi = global_row / C.TILING;
                int bj = global_col / C.TILING;
                
                // Check if this coordinate falls inside a structural nonzero block
                if (C.is_dense(bi, bj)) {
                    // Find the local offset within the BCSR tile
                    int ti = global_row % C.TILING;
                    int tj = global_col % C.TILING;
                    
                    // Use the helper method to navigate the row-interleaved layout
                    // get_tile_row returns a pointer to the start of the tile's row
                    C.get_tile_row(bi, bj, ti)[tj] = sC[warp_id][local_r][local_c];
                }
            }
        }

        // if (tile_row == 0 && tile_col == 0 && lane_id == 0) {
        //     // For debugging: print the first tile's output
        //     printf("Tile (0,0) output:\n");
        //     for (int r = 0; r < TILE_H_sddmm; r++) {
        //         for (int c = 0; c < TILE_W_sddmm; c++) {
        //             printf("%.2f ", sC[warp_id][r][c]);
        //         }
        //         printf("\n");
        //     }
        // }
    }
}
*/

static constexpr int NUM_BLOCKS_sddmm = 2048;

/// Host launcher for the SDDMM kernel.
void sddmm(float* A, float* B, BCSRMatrix& C, int M, int N, int K) {
    assert(C.get_M() == M && C.get_N() == K);

    // allocate the dynamic tile counter
    int* d_tile_counter;
    cudaMalloc(&d_tile_counter, sizeof(int));
    cudaMemset(d_tile_counter, 0, sizeof(int));
    
    BCSRView viewC = C.get_view();
    int T = viewC.TILING;
    
    char label[64];
    snprintf(label, sizeof(label), "sddmm %dx%dx%d (T=%d)", M, N, K, T);
    if (T > 16) { // 32x32 SDDMM tiling
        static constexpr int TILE_H_sddmm = 32;
        static constexpr int TILE_W_sddmm = 32;
        static constexpr int THREADS_PER_BLOCK_sddmm = 128; 
        static constexpr int WARPS_PER_BLOCK_sddmm = THREADS_PER_BLOCK_sddmm / 32;

        int shared_mem_per_block = WARPS_PER_BLOCK_sddmm * TILE_H_sddmm * TILE_W_sddmm * sizeof(float) * 4 + WARPS_PER_BLOCK_sddmm * sizeof(int);
        cudaFuncSetAttribute(sddmm_bcsr_kernel<TILE_H_sddmm, TILE_W_sddmm, WARPS_PER_BLOCK_sddmm>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_per_block);
        dim3 blockSize(THREADS_PER_BLOCK_sddmm);
        dim3 gridSize(NUM_BLOCKS_sddmm);
        time_and_print(label, [&]{ sddmm_bcsr_kernel<TILE_H_sddmm, TILE_W_sddmm, WARPS_PER_BLOCK_sddmm><<<gridSize, blockSize, shared_mem_per_block>>>(A, B, viewC, M, N, K, d_tile_counter); });
    }
    else { // 16x16 SDDMM tiling
        static constexpr int TILE_H_sddmm = 16;
        static constexpr int TILE_W_sddmm = 16;
        static constexpr int THREADS_PER_BLOCK_sddmm = 256; 
        static constexpr int WARPS_PER_BLOCK_sddmm = THREADS_PER_BLOCK_sddmm / 32;
        int shared_mem_per_block = WARPS_PER_BLOCK_sddmm * TILE_H_sddmm * TILE_W_sddmm * sizeof(float) * 4 + WARPS_PER_BLOCK_sddmm * sizeof(int);
        cudaFuncSetAttribute(sddmm_bcsr_kernel<TILE_H_sddmm, TILE_W_sddmm, WARPS_PER_BLOCK_sddmm>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_per_block);
        dim3 blockSize(THREADS_PER_BLOCK_sddmm);
        // VERSION 2: dim3 gridSize((viewC.nnzb * 32 + blockSize.x - 1) / blockSize.x);
        dim3 gridSize(NUM_BLOCKS_sddmm);
        time_and_print(label, [&]{ sddmm_bcsr_kernel<TILE_H_sddmm, TILE_W_sddmm, WARPS_PER_BLOCK_sddmm><<<gridSize, blockSize, shared_mem_per_block>>>(A, B, viewC, M, N, K, d_tile_counter); });
    }
}