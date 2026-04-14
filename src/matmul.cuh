#pragma once

/// Matrix multiplication kernel: output = A x B
/// A: M x N, B: N x K, output: M x K
/// Kernel of size at least (M, K) to cover all output elements
__global__ void matmul_kernel(float* A, float* B, float* output, int M, int N, int K);
