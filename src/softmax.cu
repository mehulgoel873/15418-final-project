#include "softmax.cuh"

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
    dim3 blockSize(16, 16);
    dim3 gridSize((row_len + blockSize.x - 1) / blockSize.x, (num_rows + blockSize.y - 1) / blockSize.y);
    softmax_kernel<<<gridSize, blockSize>>>(input, output, row_len, num_rows);
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

    // One block per output row, with chunking logic if row_len > BLOCK_WIDTH
    dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridSize(1, num_rows);
    softmax_tiled_kernel<<<gridSize, blockSize, (blockSize.x / 32) * sizeof(float)>>>(input, output, row_len, num_rows);
}