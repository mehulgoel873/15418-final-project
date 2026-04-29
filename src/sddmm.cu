#include "matmul.cuh"
#include "datastructures/bcsr.cuh"
#include "timing.cuh"
#include <cassert>
#include <cstdio>
#include <algorithm>

// SDDMM: D = (A * B^T) hadamard I[mask], where the mask is the BCSR sparsity
// pattern of `D`. A is I x K row-major, B is J x K row-major (so row j of B is
// the j-th column of B^T). For each sampled tile (bi, bj), compute the dense
// TILING x TILING product A[bi*T:(bi+1)*T, :] @ B[bj*T:(bj+1)*T, :]^T.
//
// One templated kernel parameterized by TILING; the host dispatches to the
// right specialization. Each thread block handles one row-chunk of A and a
// batch of BATCH sampled column-blocks. The B rows are gathered into a packed
// shared-mem buffer, so the inner K-loop is a clean dense GEMM with no mask
// checks.

constexpr int SDDMM_TILE_K_STEP = 32;

template <int TILING, int BATCH>
__global__ void sddmm_bcsr_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    BCSR D, int K)
{
    constexpr int NTHREADS = TILING * BATCH;

    const int bi          = blockIdx.y;
    const int batch_idx   = blockIdx.x;
    const int batch_start = D.row_ptr[bi] + batch_idx * BATCH;
    const int batch_end   = min(batch_start + BATCH, D.row_ptr[bi + 1]);
    const int actual_batch_size = batch_end - batch_start;

    if (actual_batch_size <= 0) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int linear_tid = ty * blockDim.x + tx;

    __shared__ float sA[TILING][SDDMM_TILE_K_STEP];
    __shared__ float sB[BATCH * TILING][SDDMM_TILE_K_STEP];
    __shared__ int   batch_bjs[BATCH];

    // Step 2: gather BATCH column-block indices (sentinel -1 for partial tail).
    if (linear_tid < BATCH) {
        int slot_offset = batch_start + linear_tid;
        batch_bjs[linear_tid] = (slot_offset < batch_end)
                                ? D.col_idx[slot_offset]
                                : -1;
    }
    __syncthreads();

    // Step 3: this thread's identity.
    const int my_b      = tx / TILING;
    const int my_bj     = batch_bjs[my_b];
    const int my_ti     = ty;
    const int my_tj     = tx % TILING;
    const bool my_active = (my_bj >= 0);

    float acc = 0.0f;

    const int num_k_steps = K / SDDMM_TILE_K_STEP;

    for (int t = 0; t < num_k_steps; t++) {
        const int k_base = t * SDDMM_TILE_K_STEP;

        // Load A: TILING * TILE_K_STEP elements. Each thread loads at most one.
        if (linear_tid < TILING * SDDMM_TILE_K_STEP) {
            int a_row = linear_tid / SDDMM_TILE_K_STEP;
            int a_col = linear_tid % SDDMM_TILE_K_STEP;
            sA[a_row][a_col] = A[(bi * TILING + a_row) * K + k_base + a_col];
        }

        // Gather B: BATCH*TILING rows × TILE_K_STEP cols. Thread (ty, tx) fills
        // sB[tx][s + ty] for s = 0, blockDim.y, 2*blockDim.y, ...
        const int sB_row          = tx;
        const int b_block_idx     = sB_row / TILING;
        const int b_within_row    = sB_row % TILING;
        const int bj_for_this_row = batch_bjs[b_block_idx];

        if (bj_for_this_row >= 0) {
            const int b_global_row = bj_for_this_row * TILING + b_within_row;
            #pragma unroll
            for (int s = 0; s < SDDMM_TILE_K_STEP; s += TILING) {
                sB[sB_row][s + ty] = B[b_global_row * K + k_base + s + ty];
            }
        } else {
            #pragma unroll
            for (int s = 0; s < SDDMM_TILE_K_STEP; s += TILING) {
                sB[sB_row][s + ty] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < SDDMM_TILE_K_STEP; j++) {
            acc += sA[my_ti][j] * sB[tx][j];
        }

        __syncthreads();
    }

    if (my_active) {
        D.get_tile_row(bi, my_bj, my_ti)[my_tj] = acc;
    }

    // Suppress unused-warning for NTHREADS in some configs.
    (void)NTHREADS;
}

// CPU reference: straight quadruply-nested loop. Walks D's sparsity pattern
// and computes each sampled tile's dense product. Authoritative oracle for
// the GPU kernel. D's values must be writable from host (BCSR uses
// cudaMallocManaged, so this is fine).
void sddmm_cpu(const float* A, const float* B, BCSR& D, int K) {
    const int T = D.TILING;
    for (int bi = 0; bi < D.num_block_rows; bi++) {
        for (int idx = D.row_ptr[bi]; idx < D.row_ptr[bi + 1]; idx++) {
            int bj = D.col_idx[idx];
            for (int ti = 0; ti < T; ti++) {
                float* dst = D.get_tile_row(bi, bj, ti);
                int row = bi * T + ti;
                for (int tj = 0; tj < T; tj++) {
                    int col = bj * T + tj;
                    float acc = 0.0f;
                    for (int k = 0; k < K; k++) {
                        acc += A[row * K + k] * B[col * K + k];
                    }
                    dst[tj] = acc;
                }
            }
        }
    }
}

template <int TILING, int BATCH>
static void launch_sddmm(const float* A, const float* B, BCSR& D, int K,
                         int max_per_row, const char* label) {
    int num_batches = (max_per_row + BATCH - 1) / BATCH;
    if (num_batches == 0) return;
    dim3 grid(num_batches, D.num_block_rows);
    dim3 block(BATCH * TILING, TILING);
    time_and_print(label, [&]{
        sddmm_bcsr_kernel<TILING, BATCH><<<grid, block>>>(A, B, D, K);
    });
}

// Elementwise scale of every dense value in a BCSR. Used by the transformer's
// attention path to apply 1/sqrt(d) to the post-SDDMM scores before softmax.
__global__ static void scale_bcsr_values_kernel(float* values, size_t n, float s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] *= s;
}

void scale_bcsr_values(BCSR& D, float s) {
    size_t n = (size_t)D.nnzb * D.TILING * D.TILING;
    if (n == 0) return;
    int block = 256;
    int grid  = (int)((n + block - 1) / block);
    scale_bcsr_values_kernel<<<grid, block>>>(D.values, n, s);
}

void sddmm(const float* A, const float* B, BCSR& D, int K) {
    if (K % SDDMM_TILE_K_STEP != 0) {
        fprintf(stderr,
                "sddmm: K must be divisible by %d; got K=%d\n",
                SDDMM_TILE_K_STEP, K);
        assert(false);
    }

    int max_per_row = 0;
    for (int bi = 0; bi < D.num_block_rows; bi++) {
        int n = D.row_ptr[bi + 1] - D.row_ptr[bi];
        if (n > max_per_row) max_per_row = n;
    }
    if (max_per_row == 0) return;   // all-empty mask: nothing to compute

    char label[64];
    snprintf(label, sizeof(label),
             "sddmm I=%d J=%d K=%d (T=%d)", D.M, D.N, K, D.TILING);

    cudaDeviceSynchronize();

    switch (D.TILING) {
        case 1:  launch_sddmm< 1, 256>(A, B, D, K, max_per_row, label); break;
        case 2:  launch_sddmm< 2,  64>(A, B, D, K, max_per_row, label); break;
        case 4:  launch_sddmm< 4,  16>(A, B, D, K, max_per_row, label); break;
        case 8:  launch_sddmm< 8,  16>(A, B, D, K, max_per_row, label); break;
        case 16: launch_sddmm<16,   4>(A, B, D, K, max_per_row, label); break;
        case 32: launch_sddmm<32,   1>(A, B, D, K, max_per_row, label); break;
        default:
            fprintf(stderr,
                    "sddmm: unsupported TILING=%d (must be 1,2,4,8,16,32)\n",
                    D.TILING);
            assert(false);
    }
}
