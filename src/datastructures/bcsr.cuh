#pragma once
#include <cassert>
#include <cuda_runtime.h>

// POD struct for device-side execution
struct BCSRView {
    int TILING;

    int M, N;
    int num_block_rows, num_block_cols;
    int nnzb;

    int*   block_idx;  // [num_block_rows * num_block_cols]: maps (bi,bj) -> block index in values, -1 if sparse
    int*   row_ptr;    // [num_block_rows + 1]: row_ptr[bi]..row_ptr[bi+1] indexes into col_idx for row bi
    int*   col_idx;    // [nnzb]: column block index of each dense block, in row-major order
    float* values;     // [nnzb * TILING * TILING]: row-interleaved per block-row.
                       // Block-row bi occupies values[row_ptr[bi]*T*T .. row_ptr[bi+1]*T*T),
                       // stored row-major as a dense T x (K*T) strip where K = row_ptr[bi+1]-row_ptr[bi].
                       // Row ti of the j-th dense tile (j local within the block-row) lives at
                       // values[row_ptr[bi]*T*T + ti*(K*T) + j*T .. +T).
    // nnzb: number of non-zero (dense) blocks

    /* Version 1: no matrix-view separation
     // tile_dense: flat bool array of shape [M/TILING * N/TILING], indexed by bi*num_block_cols+bj.
    // Caller is responsible for determining sparsity; we do not scan host_data ourselves.
    // host_data may be null to zero-initialize the dense tiles (useful for output buffers).
    BCSR(const float* host_data, const bool* tile_dense, int M, int N, int tiling);
    ~BCSR();

    // Shallow copy — safe for passing to CUDA kernels by value (kernel copies don't run the destructor)
    BCSR(const BCSR&)            = default;
    BCSR& operator=(const BCSR&) = delete; */

    __host__ __device__ bool is_dense(int bi, int bj) const {
        return block_idx[bi * num_block_cols + bj] >= 0;
    }

    __host__ __device__ void set_tile(int bi, int bj, bool dense) {
        block_idx[bi * num_block_cols + bj] = dense ? block_idx[bi * num_block_cols + bj] : -1;
    }

    // Number of dense tiles in block-row bi.
    __host__ __device__ int block_row_K(int bi) const {
        return row_ptr[bi + 1] - row_ptr[bi];
    }

    // Base offset (in floats) of block-row bi's strip within `values`.
    __host__ __device__ size_t block_row_base(int bi) const {
        return (size_t)row_ptr[bi] * TILING * TILING;
    }

    // Pointer to row `ti` of the tile at (bi, bj). Length TILING, stride 1
    // within the row. Stride between consecutive `ti` rows is K*TILING where
    // K = block_row_K(bi).
    __host__ __device__ float* get_tile_row(int bi, int bj, int ti) const {
        int k = block_idx[bi * num_block_cols + bj];
        // assert(k >= 0);
        int row_start = row_ptr[bi];
        int K = row_ptr[bi + 1] - row_start;
        int j = k - row_start;
        return &values[block_row_base(bi) + (size_t)ti * K * TILING + (size_t)j * TILING];
    }
};

// Host-side class managing device memory allocation and deallocation
class BCSRMatrix {
private:
    BCSRView view;

public:
    BCSRMatrix(const float* host_data, const bool* tile_dense, int M, int N, int tiling);
    ~BCSRMatrix();

    // Prevent accidental copying to avoid double-frees
    BCSRMatrix(const BCSRMatrix&) = delete;
    BCSRMatrix& operator=(const BCSRMatrix&) = delete;

    BCSRView get_view() const { return view; }

    int get_num_block_rows() const { return view.num_block_rows; }
    int get_num_block_cols() const { return view.num_block_cols; }
    int get_M() const { return view.M; }
    int get_N() const { return view.N; }
    int get_TILING() const { return view.TILING; }
};

// Backward-compatible alias for existing function signatures
using BCSR = BCSRMatrix;

// Compute the tile_dense mask for the product A*B: output tile (bi, bj) is dense
// iff there exists some bk where A(bi, bk) and B(bk, bj) are both dense.
// Returns a heap-allocated bool array of size (A.num_block_rows * B.num_block_cols);
// caller must free() it.
bool* bcsr_matmul_mask(const BCSRMatrix& A, const BCSRMatrix& B);
