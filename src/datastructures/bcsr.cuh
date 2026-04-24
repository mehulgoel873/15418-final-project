#pragma once
#include <cassert>
#include <cuda_runtime.h>

struct BCSR {
    static constexpr int TILING = 32;

    int M, N;
    int num_block_rows, num_block_cols;
    int nnzb;

    int*   block_idx;  // [num_block_rows * num_block_cols]: maps (bi,bj) -> block index in values, -1 if sparse
    int*   row_ptr;    // [num_block_rows + 1]: row_ptr[bi]..row_ptr[bi+1] indexes into col_idx for row bi
    int*   col_idx;    // [nnzb]: column block index of each dense block, in row-major order
    float* values;     // [nnzb * TILING * TILING]: packed block data; block k is values[k*TILING*TILING]
    // nnzb: number of non-zero (dense) blocks

    // tile_dense: flat bool array of shape [M/TILING * N/TILING], indexed by bi*num_block_cols+bj.
    // Caller is responsible for determining sparsity; we do not scan host_data ourselves.
    // host_data may be null to zero-initialize the dense tiles (useful for output buffers).
    BCSR(const float* host_data, const bool* tile_dense, int M, int N);
    ~BCSR();

    // Shallow copy — safe for passing to CUDA kernels by value (kernel copies don't run the destructor)
    BCSR(const BCSR&)            = default;
    BCSR& operator=(const BCSR&) = delete;

    __host__ __device__ bool is_dense(int bi, int bj) const {
        return block_idx[bi * num_block_cols + bj] >= 0;
    }

    __host__ void set_tile(int bi, int bj, bool dense) {
        block_idx[bi * num_block_cols + bj] = dense ? block_idx[bi * num_block_cols + bj] : -1;
    }

    __host__ __device__ float* get_tile(int bi, int bj) const {
        int idx = block_idx[bi * num_block_cols + bj];
        assert(idx >= 0);
        return &values[idx * TILING * TILING];
    }
};

// Compute the tile_dense mask for the product A*B: output tile (bi, bj) is dense
// iff there exists some bk where A(bi, bk) and B(bk, bj) are both dense.
// Returns a heap-allocated bool array of size (A.num_block_rows * B.num_block_cols);
// caller must free() it.
bool* bcsr_matmul_mask(const BCSR& A, const BCSR& B);
