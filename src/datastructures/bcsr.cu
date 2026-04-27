#include "bcsr.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>

BCSR::BCSR(const float* host_data, const bool* tile_dense, int M, int N, int tiling)
    : TILING(tiling), M(M), N(N),
      num_block_rows(M / tiling),
      num_block_cols(N / tiling)
{
    int T = TILING;
    int total_blocks = num_block_rows * num_block_cols;

    std::vector<int> h_block_idx(total_blocks, -1);
    std::vector<int> h_row_ptr(num_block_rows + 1, 0);
    std::vector<int> h_col_idx;
    std::vector<float> h_values;

    for (int bi = 0; bi < num_block_rows; bi++) {
        int row_start = (int)h_col_idx.size();
        h_row_ptr[bi] = row_start;

        // Pass 1: assign block indices + column-block indices for this row.
        for (int bj = 0; bj < num_block_cols; bj++) {
            if (tile_dense[bi * num_block_cols + bj]) {
                int blk = (int)h_col_idx.size();
                h_block_idx[bi * num_block_cols + bj] = blk;
                h_col_idx.push_back(bj);
            }
        }
        int K = (int)h_col_idx.size() - row_start;
        if (K == 0) continue;

        // Reserve K*T*T floats for this block-row's strip (zero-init by default).
        size_t base = (size_t)row_start * T * T;
        h_values.resize(base + (size_t)K * T * T);

        // Pass 2: copy host data into the row-interleaved strip — row ti of
        // local tile j sits at base + ti*(K*T) + j*T.
        if (host_data) {
            for (int j = 0; j < K; j++) {
                int bj = h_col_idx[row_start + j];
                for (int ti = 0; ti < T; ti++) {
                    memcpy(&h_values[base + (size_t)ti * K * T + (size_t)j * T],
                           &host_data[(bi * T + ti) * N + bj * T],
                           T * sizeof(float));
                }
            }
        }
    }
    h_row_ptr[num_block_rows] = (int)h_col_idx.size();
    nnzb = (int)h_col_idx.size();

    cudaMallocManaged(&block_idx, total_blocks * sizeof(int));
    cudaMallocManaged(&row_ptr,   (num_block_rows + 1) * sizeof(int));
    cudaMallocManaged(&col_idx,   nnzb * sizeof(int));
    cudaMallocManaged(&values,    (size_t)nnzb * T * T * sizeof(float));

    memcpy(block_idx, h_block_idx.data(), total_blocks * sizeof(int));
    memcpy(row_ptr,   h_row_ptr.data(),   (num_block_rows + 1) * sizeof(int));
    memcpy(col_idx,   h_col_idx.data(),   nnzb * sizeof(int));
    memcpy(values,    h_values.data(),    (size_t)nnzb * T * T * sizeof(float));
}

BCSR::~BCSR() {
    cudaFree(block_idx);
    cudaFree(row_ptr);
    cudaFree(col_idx);
    cudaFree(values);
}

bool* bcsr_matmul_mask(const BCSR& A, const BCSR& B) {
    assert(A.N == B.M);
    int Mb = A.num_block_rows;
    int Kb = B.num_block_cols;
    bool* mask = (bool*)calloc((size_t)Mb * Kb, sizeof(bool));
    for (int bi = 0; bi < Mb; bi++) {
        for (int k = A.row_ptr[bi]; k < A.row_ptr[bi + 1]; k++) {
            int bk = A.col_idx[k];
            for (int b = B.row_ptr[bk]; b < B.row_ptr[bk + 1]; b++) {
                int bj = B.col_idx[b];
                mask[bi * Kb + bj] = true;
            }
        }
    }
    return mask;
}
