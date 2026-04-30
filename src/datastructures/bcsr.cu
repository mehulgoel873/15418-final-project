#include "bcsr.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void bcsr_pass1_kernel(const bool* tile_dense, int num_block_rows, int num_block_cols, int* row_nnz) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi < num_block_rows) {
        int count = 0;
        for (int bj = 0; bj < num_block_cols; bj++) {
            if (tile_dense[bi * num_block_cols + bj]) {
                count++;
            }
        }
        row_nnz[bi] = count;
    }
}

__global__ void bcsr_pass2_kernel(const bool* tile_dense, int num_block_rows, int num_block_cols, 
                                  const int* row_ptr, int* block_idx, int* rev_col_idx, int* col_idx) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi < num_block_rows) {
        int start_idx = row_ptr[bi];
        int local_j = 0;
        for (int bj = 0; bj < num_block_cols; bj++) {
            int flat_idx = bi * num_block_cols + bj;
            if (tile_dense[flat_idx]) {
                int blk = start_idx + local_j;
                block_idx[flat_idx] = blk;
                rev_col_idx[flat_idx] = local_j;
                col_idx[blk] = bj;
                local_j++;
            } else {
                block_idx[flat_idx] = -1;
                rev_col_idx[flat_idx] = -1;
            }
        }
    }
}

BCSRMatrix::BCSRMatrix(const float* host_data, const bool* d_tile_dense, int M, int N, int tiling) {
    view.TILING = tiling;
    view.M = M;
    view.N = N;
    view.num_block_rows = M / tiling;
    view.num_block_cols = N / tiling;
    
    int T = view.TILING;
    int total_blocks = view.num_block_rows * view.num_block_cols;

    // Allocate device memory for structure
    // cudaMalloc instead of cudaMallocManaged to prevent page-faulting
    cudaMalloc(&view.block_idx, total_blocks * sizeof(int));
    cudaMalloc(&view.rev_col_idx, total_blocks * sizeof(int));
    cudaMalloc(&view.row_ptr, (view.num_block_rows + 1) * sizeof(int));

    // Pass 1: count nnz per row on device
    int* d_row_nnz;
    cudaMalloc(&d_row_nnz, view.num_block_rows * sizeof(int));
    
    int block_size = 256;
    int grid_size = (view.num_block_rows + block_size - 1) / block_size;
    bcsr_pass1_kernel<<<grid_size, block_size>>>(d_tile_dense, view.num_block_rows, view.num_block_cols, d_row_nnz);
    
    // Prefix sum on host
    std::vector<int> h_row_nnz(view.num_block_rows);
    cudaMemcpy(h_row_nnz.data(), d_row_nnz, view.num_block_rows * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_row_nnz);

    std::vector<int> h_row_ptr(view.num_block_rows + 1, 0);
    for(int bi = 0; bi < view.num_block_rows; bi++) {
        h_row_ptr[bi + 1] = h_row_ptr[bi] + h_row_nnz[bi];
    }
    view.nnzb = h_row_ptr[view.num_block_rows];

    cudaMemcpy(view.row_ptr, h_row_ptr.data(), (view.num_block_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    if (view.nnzb > 0) {
        cudaMalloc(&view.col_idx, view.nnzb * sizeof(int));
        
        // Reserve K*T*T floats for this block-row's strip (zero-init by default).
        cudaMalloc(&view.values, (size_t)view.nnzb * T * T * sizeof(float));
        cudaMemset(view.values, 0, (size_t)view.nnzb * T * T * sizeof(float)); // zero init values

        // Pass 2: populate sparse indices on device
        // assign block indices + column-block indices
        bcsr_pass2_kernel<<<grid_size, block_size>>>(d_tile_dense, view.num_block_rows, view.num_block_cols,
                                                     view.row_ptr, view.block_idx, view.rev_col_idx, view.col_idx);
    } else {
        view.col_idx = nullptr;
        view.values  = nullptr;
        // Fill -1 for empty matrices
        cudaMemset(view.block_idx, 0xFF, total_blocks * sizeof(int));
        cudaMemset(view.rev_col_idx, 0xFF, total_blocks * sizeof(int));
    }
}

BCSRMatrix::~BCSRMatrix() {
    cudaFree(view.block_idx);
    cudaFree(view.rev_col_idx);
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
