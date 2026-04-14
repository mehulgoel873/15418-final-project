#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void transpose_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

/// Matrix multiplication kernel: output = A x B
/// A: M x N, B: N x K, output: M x K
/// Kernel of size at least (M, K) to cover all output elements
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

/// Scaling kernel: scales the input matrix by 1/sqrt(d)
/// matrix: input/output matrix of size N x N, d: embedding dimension
__global__ void scale_kernel(float* matrix, int N, int d) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < N && col_idx < N) {
        int idx = row_idx * N + col_idx;
        matrix[idx] /= sqrtf((float)d);
    }
}

/// Softmax kernel: applies softmax to each row of the input matrix
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

class TransformerNaive {
public:
    /// Forward pass of the transformer block
    /// q, k, v: input matrices of size N x d
    /// output: output matrix of size N x d
    /// N: number of tokens, d: embedding dimension, d <= N
    void forward(float* q, float* k, float* v, float* output, int N, int d) {
        // Allocate memory for intermediate results
        float* k_transposed;
        float* attn_scores;
        float* attn_probs;
        float* attn_output;
        cudaMalloc(&k_transposed, N * d * sizeof(float));
        cudaMalloc(&attn_scores, N * N * sizeof(float));
        cudaMalloc(&attn_probs, N * N * sizeof(float));
        cudaMalloc(&attn_output, N * d * sizeof(float));

        // Compute attention scores: attn_scores = Q x K^T
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        transpose_kernel<<<gridSize, blockSize>>>(k, k_transposed, N, d);
        matmul_kernel<<<gridSize, blockSize>>>(q, k_transposed, attn_scores, N, d, N);
        cudaDeviceSynchronize();

        // Scale attention scores
        scale_kernel<<<gridSize, blockSize>>>(attn_scores, N, d);
        cudaDeviceSynchronize();

        // Apply softmax to attention scores
        softmax_kernel<<<gridSize, blockSize>>>(attn_scores, attn_probs, N, N);
        cudaDeviceSynchronize();

        // Compute attention output: attn_output = attn_probs x V
        matmul_kernel<<<gridSize, blockSize>>>(attn_probs, v, attn_output, N, N, d);
        cudaDeviceSynchronize();

        // Copy the result to the output matrix
        cudaMemcpy(output, attn_output, N * d * sizeof(float), cudaMemcpyDeviceToDevice);

        // Free allocated memory
        cudaFree(k_transposed);
        cudaFree(attn_scores);
        cudaFree(attn_probs);
        cudaFree(attn_output);
    }
};