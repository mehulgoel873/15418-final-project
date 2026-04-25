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


__global__ void matmul_sparse_bcsr_kernel(BCSR A, BCSR B, BCSR output) {
    int bi = blockIdx.y, bj = blockIdx.x;
    int out_idx = output.block_idx[bi * output.num_block_cols + bj];
    if (out_idx < 0) return; // Doestn' exist on output sparse mask

    __shared__ float sA[TILE_H][TILE_K_STEP];
    __shared__ float sB[TILE_K_STEP][TILE_W];

    int ty = threadIdx.y, tx = threadIdx.x;
    float val = 0.0f;

    int row_start = A.row_ptr[bi], row_end = A.row_ptr[bi + 1];

    for (int k_base = row_start; k_base < row_end; k_base += NUM_BATCH) {
        for (int i = 0; i < NUM_BATCH; i++) {
            int k = k_base + i;
            if (k < row_end) {
                int bk = A.col_idx[k];
                int b_idx = B.block_idx[bk * B.num_block_cols + bj];
                sA[ty][tx + i * TILE_W] = A.values[k * TILE_H * TILE_W + ty * TILE_W + tx];
                sB[ty + i * TILE_H][tx] = (b_idx >= 0)
                    ? B.values[b_idx * TILE_H * TILE_W + ty * TILE_W + tx]
                    : 0.0f;
            } else {
                sA[ty][tx + i * TILE_W] = 0.0f;
                sB[ty + i * TILE_H][tx] = 0.0f;
            }
        }
        __syncthreads();
        for (int j = 0; j < TILE_K_STEP; j++)
            val += sA[ty][j] * sB[j][tx];
        __syncthreads();
    }

    output.values[out_idx * TILE_H * TILE_W + ty * TILE_W + tx] = val;
}

void matmul_sparse_bcsr(BCSR& A, BCSR& B, BCSR& output) {
    cudaDeviceSynchronize();
    char label[64];
    snprintf(label, sizeof(label), "matmul_sparse_bcsr %dx%dx%d", A.M, A.N, B.N);
    dim3 blockSize(TILE_W, TILE_H);
    dim3 gridSize(output.num_block_cols, output.num_block_rows);
    time_and_print(label, [&]{ matmul_sparse_bcsr_kernel<<<gridSize, blockSize>>>(A, B, output); });
}


// One thread per element: mat[i,j] *= mask[i,j].
__global__ void elemwise_mul_kernel(float* mat, const float* mask, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    int idx = row * n + col;
    mat[idx] *= mask[idx];
}
