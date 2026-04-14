#pragma once

/// Host launcher: sets up grid/block dims and calls matmul_tiled_kernel.
/// Requires M % 8 == 0, K % 8 == 0, N % 8 == 0.
void matmul_tiled(float* A, float* B, float* output, int M, int N, int K);

/// Host launcher for the naive matmul kernel (16x16 blocks, no shared memory).
void matmul_naive(float* A, float* B, float* output, int M, int N, int K);
