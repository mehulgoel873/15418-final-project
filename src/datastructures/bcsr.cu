#include "bcsr.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>

BCSRMatrix::BCSRMatrix(const float* host_data, const bool* tile_dense, int M, int N, int tiling) {
    view.TILING = tiling;
    view.M = M;
    view.N = N;
    view.num_block_rows = M / tiling;
    view.num_block_cols = N / tiling;
    
    int T = view.TILING;
    int total_blocks = view.num_block_rows * view.num_block_cols;

    std::vector<int> h_block_idx(total_blocks, -1);
    std::vector<int> h_row_ptr(view.num_block_rows + 1, 0);
    std::vector<int> h_col_idx;
    std::vector<float> h_values;

    for (int bi = 0; bi < view.num_block_rows; bi++) {
        int row_start = (int)h_col_idx.size();
        h_row_ptr[bi] = row_start;

        // Pass 1: assign block indices + column-block indices for this row.
        for (int bj = 0; bj < view.num_block_cols; bj++) {
            if (tile_dense[bi * view.num_block_cols + bj]) {
                int blk = (int)h_col_idx.size();
                h_block_idx[bi * view.num_block_cols + bj] = blk;
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
    h_row_ptr[view.num_block_rows] = (int)h_col_idx.size();
    view.nnzb = (int)h_col_idx.size();

    // cudaMalloc instead of cudaMallocManaged to prevent page-faulting
    cudaMalloc(&view.block_idx, total_blocks * sizeof(int));
    cudaMalloc(&view.row_ptr,   (view.num_block_rows + 1) * sizeof(int));
    cudaMemcpy(view.block_idx, h_block_idx.data(), total_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(view.row_ptr,   h_row_ptr.data(),   (view.num_block_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    if (view.nnzb > 0) {
        cudaMalloc(&view.col_idx,   view.nnzb * sizeof(int));
        cudaMalloc(&view.values,    (size_t)view.nnzb * T * T * sizeof(float));
        cudaMemcpy(view.col_idx,   h_col_idx.data(),   view.nnzb * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(view.values,    h_values.data(),    (size_t)view.nnzb * T * T * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        view.col_idx = nullptr;
        view.values  = nullptr;
    }
}

BCSRMatrix::~BCSRMatrix() {
    cudaFree(view.block_idx);
    cudaFree(view.row_ptr);
    if (view.col_idx) cudaFree(view.col_idx);
    if (view.values)  cudaFree(view.values);
}

bool* bcsr_matmul_mask(const BCSRMatrix& A, const BCSRMatrix& B) {
    assert(A.get_N() == B.get_M());
    BCSRView vA = A.get_view();
    BCSRView vB = B.get_view();
    
    int Mb = vA.num_block_rows;
    int Kb = vB.num_block_cols;
    bool* mask = (bool*)calloc((size_t)Mb * Kb, sizeof(bool));
    
    // TODO: without unified memory, need to compute the mask on the device or retain host-side arrays.
    // temporary solution: copy back the necessary arrays to the host to build the mask
    int* hA_row_ptr = (int*)malloc((vA.num_block_rows + 1) * sizeof(int));
    int* hB_row_ptr = (int*)malloc((vB.num_block_rows + 1) * sizeof(int));
    int* hA_col_idx = (int*)malloc(vA.nnzb * sizeof(int));
    int* hB_col_idx = (int*)malloc(vB.nnzb * sizeof(int));
    
    cudaMemcpy(hA_row_ptr, vA.row_ptr, (vA.num_block_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hB_row_ptr, vB.row_ptr, (vB.num_block_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hA_col_idx, vA.col_idx, vA.nnzb * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hB_col_idx, vB.col_idx, vB.nnzb * sizeof(int), cudaMemcpyDeviceToHost);

    for (int bi = 0; bi < Mb; bi++) {
        for (int k = hA_row_ptr[bi]; k < hA_row_ptr[bi + 1]; k++) {
            int bk = hA_col_idx[k];
            for (int b = hB_row_ptr[bk]; b < hB_row_ptr[bk + 1]; b++) {
                int bj = hB_col_idx[b];
                mask[bi * Kb + bj] = true;
            }
        }
    }
    
    free(hA_row_ptr);
    free(hB_row_ptr);
    free(hA_col_idx);
    free(hB_col_idx);
    
    return mask;
}
