#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "matmul.cuh"

// Fill a host array with uniform random values in [0, 1).
static void rand_fill(float* h, int n) {
    for (int i = 0; i < n; i++)
        h[i] = (float)rand() / (float)RAND_MAX;
}

// Returns the maximum absolute difference between two host arrays.
static float max_abs_diff(const float* a, const float* b, int n) {
    float diff = 0.0f;
    for (int i = 0; i < n; i++)
        diff = fmaxf(diff, fabsf(a[i] - b[i]));
    return diff;
}

// Run matmul_naive and matmul (tiled) on the same M x N x K problem,
// copy results back to the host, and return the max absolute difference.
// Caller provides pre-filled host matrices h_A and h_B.
static float run_test(const float* h_A, const float* h_B,
                      int M, int N, int K) {
    size_t bytes_A   = (size_t)M * N * sizeof(float);
    size_t bytes_B   = (size_t)N * K * sizeof(float);
    size_t bytes_out = (size_t)M * K * sizeof(float);

    float *d_A, *d_B, *d_naive, *d_tiled;
    cudaMalloc(&d_A,     bytes_A);
    cudaMalloc(&d_B,     bytes_B);
    cudaMalloc(&d_naive, bytes_out);
    cudaMalloc(&d_tiled, bytes_out);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    matmul_naive(d_A, d_B, d_naive, M, N, K);
    matmul_tiled(d_A, d_B, d_tiled, M, N, K);
    cudaDeviceSynchronize();

    float* h_naive = (float*)malloc(bytes_out);
    float* h_tiled = (float*)malloc(bytes_out);
    cudaMemcpy(h_naive, d_naive, bytes_out, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tiled, d_tiled, bytes_out, cudaMemcpyDeviceToHost);

    float diff = max_abs_diff(h_naive, h_tiled, M * K);

    free(h_naive);
    free(h_tiled);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_naive);
    cudaFree(d_tiled);

    return diff;
}

struct TestCase { int M, N, K; };

int main() {
    srand(42);

    // All dimensions must be multiples of 8 (tiled kernel requirement).
    TestCase cases[] = {
        {  8,   8,   8 },   // minimal: single tile
        { 16,   8,  16 },   // 2x1 grid of tiles
        { 32,  32,  32 },   // square, several tiles
        { 64, 128,  64 },   // non-square M/K, larger N
        {256,  64, 128 },   // rectangular, stress the grid sizing
    };

    const float kTol = 1e-3f;   // float32 accumulation error grows with N
    int passed = 0, failed = 0;

    printf("%-20s  %-20s  %12s  %s\n", "Shape (MxNxK)", "", "Max |diff|", "Result");
    printf("%-20s  %-20s  %12s  %s\n", "--------------------",
           "--------------------", "------------", "------");

    for (auto& tc : cases) {
        int M = tc.M, N = tc.N, K = tc.K;

        float* h_A = (float*)malloc((size_t)M * N * sizeof(float));
        float* h_B = (float*)malloc((size_t)N * K * sizeof(float));
        rand_fill(h_A, M * N);
        rand_fill(h_B, N * K);

        float diff = run_test(h_A, h_B, M, N, K);

        free(h_A);
        free(h_B);

        bool ok = diff < kTol;
        ok ? passed++ : failed++;
        printf("%dx%dx%-14d  max|diff|=%12.2e  %s\n",
               M, N, K, diff, ok ? "PASS" : "FAIL");
    }

    printf("\n%d/%zu passed\n", passed, sizeof(cases) / sizeof(cases[0]));
    return failed > 0 ? 1 : 0;
}
