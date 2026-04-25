#include "softmax.cuh"
#include "datastructures/bcsr.cuh"
#include "timing.cuh"
#include <cstdio>
#include <cstring>
#include <vector>

/// Naive softmax kernel: output = row-wise softmax(input)
/// input: row_len x num_rows, output: row_len x num_rows
__global__ void softmax_kernel(float* input, float* output, int row_len, int num_rows) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < num_rows && col_idx < row_len) {
        int row_start = row_idx * row_len;
        int idx = row_start + col_idx;

        // Subtract the maximum value for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < row_len; i++) {
            max_val = fmaxf(max_val, input[row_start + i]);
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < row_len; i++) {
            sum_exp += expf(input[row_start + i] - max_val);
        }
        output[idx] = expf(input[idx] - max_val) / sum_exp;
    }
}

    
/// Host launcher for the naive softmax kernel.
void softmax_naive(float* input, float* output, int row_len, int num_rows) {
    char label[64];
    snprintf(label, sizeof(label), "softmax_naive %dx%d", num_rows, row_len);
    dim3 blockSize(16, 16);
    dim3 gridSize((row_len + blockSize.x - 1) / blockSize.x, (num_rows + blockSize.y - 1) / blockSize.y);
    time_and_print(label, [&]{ softmax_kernel<<<gridSize, blockSize>>>(input, output, row_len, num_rows); });
}


// ---------------------------------------------------------------------------
// Tiled softmax - blocked along the row dimension
// Uses __shfl_sync for warp-level reduction, 
//      shared memory for block-wide reduction, 
//      chunk processing to handle large rows
//
// blockDim = BLOCK_WIDTH=min(row_len, 1024), BLOCK_HEIGHT=1
// Each output row is processed by 1 block, with chunking if row_len > BLOCK_WIDTH
// Shared memory per block: BLOCK_WIDTH floats for reduction (max and sum) <= 1024 * 4 bytes = 4KB
//
// ---------------------------------------------------------------------------

__global__ void softmax_tiled_kernel(float* input, float* output, int row_len, int num_rows) {
    extern __shared__ float shared_mem[];

    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx >= num_rows) return;

    // Each block processes one row, potentially in multiple chunks if row_len > BLOCK_WIDTH
    int row_start = row_idx * row_len;

    // Step 1: Compute the max value for numerical stability using parallel reduction
    float max_val = -INFINITY;
    for (int col = threadIdx.x; col < row_len; col += blockDim.x) {
        if (col < row_len) {
            max_val = fmaxf(max_val, input[row_start + col]);
        }
    }

    // implicit __syncwarp() here due to subsequent __shfl_sync call
    // Warp-level reduction using __shfl_sync (mask can be 0xFFFFFFFF since any non-active threads have max_val=-INFINITY)
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }

    // Write warp maxes to shared memory for block-wide reduction
    if (threadIdx.x % 32 == 0) { // one thread per warp
        shared_mem[threadIdx.x / 32] = max_val; // store warp maxes
    }
    __syncthreads();

    // Warp 0 Sweep: Perform parallel reduction in shared memory using Warp 0
    // Warp 0 Sweep helps handle cases where blockDim.x is not a power of two
    if (threadIdx.x < 32) { // only first warp participates in reduction
        max_val = threadIdx.x < blockDim.x / 32 ? shared_mem[threadIdx.x] : -INFINITY; // load warp maxes into registers

        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
        }
        if (threadIdx.x == 0) {
            shared_mem[0] = max_val; // store final max in shared memory
        }
    }
    __syncthreads();
    max_val = shared_mem[0];


    /* for (int stride = blockDim.x / 2 / 32; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] = fmaxf(shared_mem[threadIdx.x], shared_mem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    max_val = shared_mem[0]; */

    // Step 2: Compute the sum of exponentials, using __shfl_sync for warp-level reduction and shared memory for block-wide reduction
    float exp_val = 0.0f;
    float sum_exp = 0.0f;
    for (int col = threadIdx.x; col < row_len; col += blockDim.x) {
        if (col < row_len) {
            exp_val = expf(input[row_start + col] - max_val);
            output[row_start + col] = exp_val; // store exp values temporarily in output
            sum_exp += exp_val;
        }
    }

    // implicit __syncwarp() here due to subsequent __shfl_sync call
    // Warp-level reduction using __shfl_sync (mask can be 0xFFFFFFFF since any non-active threads have sum_exp=0)
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }

    // Write warp sums to shared memory for block-wide reduction
    if (threadIdx.x % 32 == 0) { // one thread per warp
        shared_mem[threadIdx.x / 32] = sum_exp; // store warp sums
    }
    __syncthreads();

    // Warp 0 Sweep: Perform parallel reduction of warp sums in shared memory using Warp 0
    if (threadIdx.x < 32) { // only first warp participates in reduction
        sum_exp = threadIdx.x < blockDim.x / 32 ? shared_mem[threadIdx.x] : 0.0f; // load warp sums into registers

        for (int offset = 16; offset > 0; offset /= 2) {
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        }
        if (threadIdx.x == 0) {
            shared_mem[0] = sum_exp; // store final sum in shared memory
        }
    }
    __syncthreads();
    sum_exp = shared_mem[0];


/*     for (int stride = blockDim.x / 2 / 32; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    sum_exp = shared_mem[0]; */

    // Step 3: Write the final softmax output
    for (int col = threadIdx.x; col < row_len; col += blockDim.x) {
        if (col < row_len) {
            output[row_start + col] /= sum_exp; // normalize by sum of exponentials
        }
    }

}

/// Host launcher for the tiled softmax.
void softmax_tiled(float* input, float* output, int row_len, int num_rows) {
    int BLOCK_WIDTH = ((min(1024, row_len) + 31) / 32) * 32; // Round up to nearest multiple of 32 to prevent warp sync deadlocks
    static constexpr int BLOCK_HEIGHT = 1;

    char label[64];
    snprintf(label, sizeof(label), "softmax_tiled %dx%d", num_rows, row_len);
    // One block per output row, with chunking logic if row_len > BLOCK_WIDTH
    dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridSize(1, num_rows);
    size_t smem = (blockSize.x / 32) * sizeof(float);
    time_and_print(label, [&]{ softmax_tiled_kernel<<<gridSize, blockSize, smem>>>(input, output, row_len, num_rows); });
}


// Alternative Approach: Dynamic CUDA-managed scheduling
// Assigns one block per row dynamically using an atomic counter to guarantee load balancing.
__global__ void softmax_bcsr_bcsr_kernel_dynamic(BCSR input, BCSR output, int* row_counter) {
    extern __shared__ float shared_mem[];
    __shared__ int s_row;

    int T = input.TILING;

    while (true) {
        if (threadIdx.x == 0) s_row = atomicAdd(row_counter, 1);
        __syncthreads();
        
        int row = s_row;
        if (row >= input.M) break;

        int bi = row / T;
        int t = row % T;
        int blk_start = input.row_ptr[bi];
        int num_blks = input.row_ptr[bi + 1] - blk_start;
        if (num_blks == 0) continue;

        int N_elem = num_blks * T;
        
        // Phase 1: Max reduction (shared memory + warp reduction)
        float max_val = -INFINITY;
        for (int idx = threadIdx.x; idx < N_elem; idx += blockDim.x) {
            int blk = blk_start + (idx / T);
            int c = idx % T;
            max_val = fmaxf(max_val, input.values[blk * T * T + t * T + c]);
        }
        for (int offset = 16; offset > 0; offset /= 2) max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
        if (threadIdx.x % 32 == 0) shared_mem[threadIdx.x / 32] = max_val;
        __syncthreads();
        if (threadIdx.x < 32) {
            max_val = threadIdx.x < (blockDim.x / 32) ? shared_mem[threadIdx.x] : -INFINITY;
            for (int offset = 16; offset > 0; offset /= 2) {
                max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
            }
            if (threadIdx.x == 0) {
                shared_mem[0] = max_val;
            }
        }
        __syncthreads();
        max_val = shared_mem[0];

        // Phase 2: Exp sum & write
        float sum_exp = 0.0f;
        for (int idx = threadIdx.x; idx < N_elem; idx += blockDim.x) {
            int blk = blk_start + (idx / T);
            int c = idx % T;
            float e = expf(input.values[blk * T * T + t * T + c] - max_val);
            int out_idx = output.block_idx[bi * output.num_block_cols + input.col_idx[blk]];
            output.values[out_idx * T * T + t * T + c] = e;
            sum_exp += e;
        }
        for (int offset = 16; offset > 0; offset /= 2) sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        if (threadIdx.x % 32 == 0) shared_mem[threadIdx.x / 32] = sum_exp;
        __syncthreads();
        if (threadIdx.x < 32) {
            sum_exp = threadIdx.x < (blockDim.x / 32) ? shared_mem[threadIdx.x] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
            }
            if (threadIdx.x == 0) {
                shared_mem[0] = sum_exp;
            }
        }
        __syncthreads();
        sum_exp = shared_mem[0];

        // Phase 3: Normalize
        for (int idx = threadIdx.x; idx < N_elem; idx += blockDim.x) {
            int blk = blk_start + (idx / T);
            int c = idx % T;
            int out_idx = output.block_idx[bi * output.num_block_cols + input.col_idx[blk]];
            output.values[out_idx * T * T + t * T + c] /= sum_exp;
        }
        __syncthreads(); // before processing next row in same block
    }
}

__global__ void softmax_bcsr_bcsr_kernel(BCSR input, BCSR output, const int* row_partitions) {
    extern __shared__ float shared_mem[];

    int T = input.TILING;
    int row_start = row_partitions[blockIdx.x];
    int row_end   = row_partitions[blockIdx.x + 1];

    for (int row = row_start; row < row_end; row++) {
        int bi = row / T;
        int t = row % T;
        
        int blk_start = input.row_ptr[bi];
        int num_blks = input.row_ptr[bi + 1] - blk_start;
        if (num_blks == 0) continue;

        int N_elem = num_blks * T;

        // Phase 1: Max
        float max_val = -INFINITY;
        for (int idx = threadIdx.x; idx < N_elem; idx += blockDim.x) {
            int blk = blk_start + (idx / T);
            int c = idx % T;
            max_val = fmaxf(max_val, input.values[blk * T * T + t * T + c]);
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
        }

        if (threadIdx.x % 32 == 0) {
            shared_mem[threadIdx.x / 32] = max_val;
        }
        __syncthreads();

        if (threadIdx.x < 32) {
            max_val = threadIdx.x < blockDim.x / 32 ? shared_mem[threadIdx.x] : -INFINITY;
            for (int offset = 16; offset > 0; offset /= 2) {
                max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
            }
            if (threadIdx.x == 0) {
                shared_mem[0] = max_val;
            }
        }
        __syncthreads();
        max_val = shared_mem[0];

        // Phase 2: Sum Exp
        float sum_exp = 0.0f;
        for (int idx = threadIdx.x; idx < N_elem; idx += blockDim.x) {
            int blk = blk_start + (idx / T);
            int c = idx % T;
            int out_idx = output.block_idx[bi * output.num_block_cols + input.col_idx[blk]];
            float e = expf(input.values[blk * T * T + t * T + c] - max_val);
            output.values[out_idx * T * T + t * T + c] = e; 
            sum_exp += e;
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        }

        if (threadIdx.x % 32 == 0) {
            shared_mem[threadIdx.x / 32] = sum_exp;
        }
        __syncthreads();

        if (threadIdx.x < 32) {
            sum_exp = threadIdx.x < blockDim.x / 32 ? shared_mem[threadIdx.x] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
            }
            if (threadIdx.x == 0) {
                shared_mem[0] = sum_exp;
            }
        }
        __syncthreads();
        sum_exp = shared_mem[0];

        // Phase 3: Normalize
        for (int idx = threadIdx.x; idx < N_elem; idx += blockDim.x) {
            int blk = blk_start + (idx / T);
            int c = idx % T;
            int out_idx = output.block_idx[bi * output.num_block_cols + input.col_idx[blk]];
            output.values[out_idx * T * T + t * T + c] /= sum_exp;
        }
        __syncthreads(); // before processing next row in same block
    }
}

void softmax_bcsr_bcsr(BCSR& input, BCSR& output) {
    cudaDeviceSynchronize();
    
    int num_blocks = 256;
    int BLOCK_WIDTH = 256;
    dim3 blockSize(BLOCK_WIDTH, 1);
    dim3 gridSize(num_blocks, 1);
    size_t smem = (BLOCK_WIDTH / 32) * sizeof(float);

    if (SOFTMAX_BCSR_USE_DYNAMIC) {
        int* d_row_counter;
        cudaMalloc(&d_row_counter, sizeof(int));
        cudaMemset(d_row_counter, 0, sizeof(int));

        char label[64];
        snprintf(label, sizeof(label), "softmax_bcsr_bcsr %dx%d (dynamic tiled)", input.M, input.N);
        
        time_and_print(label, [&]{ 
            softmax_bcsr_bcsr_kernel_dynamic<<<gridSize, blockSize, smem>>>(input, output, d_row_counter); 
        });

        cudaFree(d_row_counter);
    } else {
        int M = input.M;
        int T = input.TILING;

        // Weight metric: number of non-zero elements per row
        int total_weight = 0;
        std::vector<int> row_weights(M, 0);
        for (int row = 0; row < M; row++) {
            int bi = row / T;
            int w = input.row_ptr[bi + 1] - input.row_ptr[bi]; // dense blocks
            row_weights[row] = w;
            total_weight += w;
        }

        std::vector<int> partitions(num_blocks + 1, 0);
        int target_weight = (total_weight + num_blocks - 1) / num_blocks; // ceil div
        int current_weight = 0;
        int b_idx = 1;

        for (int row = 0; row < M; row++) {
            current_weight += row_weights[row];
            if (current_weight >= target_weight && b_idx < num_blocks) {
                partitions[b_idx++] = row + 1;
                current_weight = 0;
            }
        }
        while (b_idx <= num_blocks) partitions[b_idx++] = M;

        int* d_partitions;
        cudaMalloc(&d_partitions, (num_blocks + 1) * sizeof(int));
        cudaMemcpy(d_partitions, partitions.data(), (num_blocks + 1) * sizeof(int), cudaMemcpyHostToDevice);

        char label[64];
        snprintf(label, sizeof(label), "softmax_bcsr_bcsr %dx%d (static tiled)", input.M, input.N);
        
        time_and_print(label, [&]{ 
            softmax_bcsr_bcsr_kernel<<<gridSize, blockSize, smem>>>(input, output, d_partitions); 
        });

        cudaFree(d_partitions);
    }
}