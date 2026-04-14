#include "matmul.cuh"
#include <cassert>
#include <cstdio>

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
    dim3 blockSize(16, 16);
    dim3 gridSize((K + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<gridSize, blockSize>>>(A, B, output, M, N, K);
}


// ---------------------------------------------------------------------------
// Tiled matrix multiplication — 8x8 output tile, 2 elements per thread
//
// blockDim = (TILE_WIDTH=8, TILE_HEIGHT=4) -> 32 threads = one full warp.
//
// Each block owns an 8x8 region of the output.  With only 4 rows of threads,
// each thread computes two output elements: rows threadIdx.y and threadIdx.y+4.
//
// Inner tile step: TILE_K = 8 (columns of A / rows of B loaded per iteration).
//
// Shared memory per block (128 floats = 512 bytes, well within limits):
//   sA[8][8]  — 8 rows x 8 cols of A
//   sB[8][8]  — 8 rows x 8 cols of B
//
// Loading: both tiles have 64 elements and we have 32 threads, so every
// thread does exactly 2 loads for sA and 2 loads for sB — fully symmetric,
// no conditionals, all threads active.
// ---------------------------------------------------------------------------

static constexpr int TILE_HEIGHT = 4;   // threads in y; each covers 2 output rows
static constexpr int TILE_WIDTH  = 8;   // threads in x; output tile width
static constexpr int TILE_K      = 8;   // inner tile step (= TILE_WIDTH -> square shared tiles)

__global__ void matmul_tiled_kernel(float* A, float* B, float* output, int M, int N, int K) {
    __shared__ float sA[TILE_K][TILE_WIDTH];   // [8][8], slice of A
    __shared__ float sB[TILE_K][TILE_WIDTH];   // [8][8], slice of B

    // The two output rows this thread is responsible for.
    int row0 = blockIdx.y * (TILE_HEIGHT * 2) + threadIdx.y;
    int row1 = row0 + TILE_HEIGHT;
    int col  = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float val0 = 0.0f;   // accumulator for row0
    float val1 = 0.0f;   // accumulator for row1

    // Flat index of this thread within the 32-thread block (0..31).
    int lin = threadIdx.y * TILE_WIDTH + threadIdx.x;

    int numTiles = N / TILE_K;
    for (int t = 0; t < numTiles; t++) {

        // --- Load sA[8][8] (64 elements, 2 per thread) ---------------
        int a0r = lin / TILE_WIDTH,        a0c = lin % TILE_WIDTH;
        int a1r = (lin + 32) / TILE_WIDTH, a1c = (lin + 32) % TILE_WIDTH;
        sA[a0r][a0c] = A[(blockIdx.y * (TILE_HEIGHT * 2) + a0r) * N + t * TILE_K + a0c];
        sA[a1r][a1c] = A[(blockIdx.y * (TILE_HEIGHT * 2) + a1r) * N + t * TILE_K + a1c];

        // --- Load sB[8][8] (64 elements, 2 per thread) ---------------
        int b0r = lin / TILE_WIDTH,        b0c = lin % TILE_WIDTH;
        int b1r = (lin + 32) / TILE_WIDTH, b1c = (lin + 32) % TILE_WIDTH;
        sB[b0r][b0c] = B[(t * TILE_K + b0r) * K + blockIdx.x * TILE_WIDTH + b0c];
        sB[b1r][b1c] = B[(t * TILE_K + b1r) * K + blockIdx.x * TILE_WIDTH + b1c];

        __syncthreads();

        // --- Accumulate partial dot products --------------------------
        // Both output elements share the same sB column (threadIdx.x) but
        // read different rows of sA: threadIdx.y for val0, threadIdx.y+4 for val1.
        for (int j = 0; j < TILE_K; j++) {
            val0 += sA[threadIdx.y][j]              * sB[j][threadIdx.x];
            val1 += sA[threadIdx.y + TILE_HEIGHT][j] * sB[j][threadIdx.x];
        }

        __syncthreads();
    }

    output[row0 * K + col] = val0;
    output[row1 * K + col] = val1;
}

/// Host launcher for the tiled matmul.
/// Requires M % 8 == 0, K % 8 == 0, N % 8 == 0.
void matmul_tiled(float* A, float* B, float* output, int M, int N, int K) {
    if (M % (TILE_HEIGHT * 2) != 0 || K % TILE_WIDTH != 0 || N % TILE_K != 0) {
        fprintf(stderr,
                "matmul: dimensions must be divisible by tile sizes "
                "(M%%%d, N%%%d, K%%%d); got M=%d N=%d K=%d\n",
                TILE_HEIGHT * 2, TILE_K, TILE_WIDTH, M, N, K);
        assert(false);
    }

    // One block per 8x8 output tile; each block is exactly one warp (32 threads).
    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);               // (8, 4)
    dim3 gridSize(K / TILE_WIDTH, M / (TILE_HEIGHT * 2));  // covers the full MxK output
    matmul_tiled_kernel<<<gridSize, blockSize>>>(A, B, output, M, N, K);
}