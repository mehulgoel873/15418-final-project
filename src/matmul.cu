#include "matmul.cuh"
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


// ---------------------------------------------------------------------------
// Tiled matrix multiplication — 32x32 output tile, 1 element per thread
//
// blockDim = (32, 32) = 1024 threads = the per-block maximum.
// Each thread owns exactly one output element: output[row][col].
//
// Inner tile step: TILE_K = 32 (columns of A / rows of B per iteration).
//
// Shared memory per block:
//   sA[32][32] = 4 KB  (TILE_HEIGHT rows x TILE_K cols of A)
//   sB[32][32] = 4 KB  (TILE_K rows x TILE_WIDTH cols of B)
//   Total: 8 KB — well within the 48 KB limit.
//
// Loading: 1024 threads, 1024 elements in each tile -> 1 load per thread,
// no conditionals, no idle threads, perfectly symmetric.
//
// Arithmetic intensity: T/4 = 32/4 = 8 FLOPs/byte (vs 2 for the old 8x8 tile).
// ---------------------------------------------------------------------------

static constexpr int TILE_HEIGHT = 32;
static constexpr int TILE_WIDTH  = 32;
static constexpr int TILE_K      = 32;

__global__ void matmul_tiled_kernel(float* A, float* B, float* output, int M, int N, int K) {
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
}

/// Host launcher for the tiled matmul.
/// Requires M % 32 == 0, K % 32 == 0, N % 32 == 0.
void matmul_tiled(float* A, float* B, float* output, int M, int N, int K) {
    if (M % TILE_HEIGHT != 0 || K % TILE_WIDTH != 0 || N % TILE_K != 0) {
        fprintf(stderr,
                "matmul_tiled: dimensions must be divisible by %d; got M=%d N=%d K=%d\n",
                TILE_HEIGHT, M, N, K);
        assert(false);
    }

    char label[64];
    snprintf(label, sizeof(label), "matmul_tiled %dx%dx%d", M, N, K);
    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize(K / TILE_WIDTH, M / TILE_HEIGHT);
    time_and_print(label, [&]{ matmul_tiled_kernel<<<gridSize, blockSize>>>(A, B, output, M, N, K); });
}