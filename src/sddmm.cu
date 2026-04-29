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
    BCSRView D, int K)
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
// the GPU kernel. BCSR is now backed by raw device memory (no unified memory),
// so we copy row_ptr / col_idx to host, compute values into a host buffer, and
// memcpy values back to the device.
void sddmm_cpu(const float* A, const float* B, BCSR& D, int K) {
    BCSRView dv = D.get_view();
    const int T = dv.TILING;

    int* h_row_ptr = (int*)malloc((dv.num_block_rows + 1) * sizeof(int));
    int* h_col_idx = (int*)malloc((size_t)dv.nnzb * sizeof(int));
    cudaMemcpy(h_row_ptr, dv.row_ptr,
               (dv.num_block_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx, dv.col_idx,
               (size_t)dv.nnzb * sizeof(int), cudaMemcpyDeviceToHost);

    size_t nvals = (size_t)dv.nnzb * T * T;
    float* h_values = (float*)calloc(nvals, sizeof(float));

    for (int bi = 0; bi < dv.num_block_rows; bi++) {
        int row_start = h_row_ptr[bi];
        int K_b       = h_row_ptr[bi + 1] - row_start;
        if (K_b == 0) continue;
        size_t base = (size_t)row_start * T * T;
        for (int j = 0; j < K_b; j++) {
            int bj = h_col_idx[row_start + j];
            for (int ti = 0; ti < T; ti++) {
                float* dst = &h_values[base + (size_t)ti * K_b * T + (size_t)j * T];
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

    cudaMemcpy(dv.values, h_values, nvals * sizeof(float), cudaMemcpyHostToDevice);

    free(h_row_ptr);
    free(h_col_idx);
    free(h_values);
}

template <int TILING, int BATCH>
static void launch_sddmm(const float* A, const float* B, BCSRView dv, int K,
                         int max_per_row, const char* label) {
    int num_batches = (max_per_row + BATCH - 1) / BATCH;
    if (num_batches == 0) return;
    dim3 grid(num_batches, dv.num_block_rows);
    dim3 block(BATCH * TILING, TILING);
    time_and_print(label, [&]{
        sddmm_bcsr_kernel<TILING, BATCH><<<grid, block>>>(A, B, dv, K);
    });
}

// Elementwise scale of every dense value in a BCSR. Used by the transformer's
// attention path to apply 1/sqrt(d) to the post-SDDMM scores before softmax.
__global__ static void scale_bcsr_values_kernel(float* values, size_t n, float s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] *= s;
}

void scale_bcsr_values(BCSR& D, float s) {
    BCSRView dv = D.get_view();
    size_t n = (size_t)dv.nnzb * dv.TILING * dv.TILING;
    if (n == 0) return;
    int block = 256;
    int grid  = (int)((n + block - 1) / block);
    scale_bcsr_values_kernel<<<grid, block>>>(dv.values, n, s);
}

void sddmm(const float* A, const float* B, BCSR& D, int K) {
    if (K % SDDMM_TILE_K_STEP != 0) {
        fprintf(stderr,
                "sddmm: K must be divisible by %d; got K=%d\n",
                SDDMM_TILE_K_STEP, K);
        assert(false);
    }

    BCSRView dv = D.get_view();

    // max_block_row_K is precomputed at BCSR construction (cheap host int —
    // no device round-trip per call).
    int max_per_row = dv.max_block_row_K;
    if (max_per_row == 0) return;   // all-empty mask: nothing to compute

    char label[64];
    snprintf(label, sizeof(label),
             "sddmm I=%d J=%d K=%d (T=%d)", dv.M, dv.N, K, dv.TILING);

    cudaDeviceSynchronize();

    switch (dv.TILING) {
        case 1:  launch_sddmm< 1, 256>(A, B, dv, K, max_per_row, label); break;
        case 2:  launch_sddmm< 2,  64>(A, B, dv, K, max_per_row, label); break;
        case 4:  launch_sddmm< 4,  16>(A, B, dv, K, max_per_row, label); break;
        case 8:  launch_sddmm< 8,  16>(A, B, dv, K, max_per_row, label); break;
        case 16: launch_sddmm<16,   4>(A, B, dv, K, max_per_row, label); break;
        case 32: launch_sddmm<32,   1>(A, B, dv, K, max_per_row, label); break;
        default:
            fprintf(stderr,
                    "sddmm: unsupported TILING=%d (must be 1,2,4,8,16,32)\n",
                    dv.TILING);
            assert(false);
    }
}
