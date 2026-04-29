#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "matmul.cuh"
#include "datastructures/bcsr.cuh"

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

// Run matmul_naive on the full dense A vs matmul_sparse on the BCSR version of A.
// A sparse tile is zeroed out in both the dense matrix and the BCSR mask so both
// kernels operate on the same mathematical matrix.
static float run_sparse_test(int M, int N, int K, float sparsity) {
    size_t bytes_A   = (size_t)M * N * sizeof(float);
    size_t bytes_B   = (size_t)N * K * sizeof(float);
    size_t bytes_out = (size_t)M * K * sizeof(float);

    constexpr int TILING = 32;
    int num_block_rows = M / TILING;
    int num_block_cols = N / TILING;
    int T = TILING;

    float* h_A = (float*)calloc(M * N, sizeof(float));
    float* h_B = (float*)malloc(bytes_B);
    rand_fill(h_B, N * K);

    bool* tile_dense = (bool*)malloc(num_block_rows * num_block_cols * sizeof(bool));
    for (int bi = 0; bi < num_block_rows; bi++) {
        for (int bj = 0; bj < num_block_cols; bj++) {
            bool dense = ((float)rand() / RAND_MAX) >= sparsity;
            tile_dense[bi * num_block_cols + bj] = dense;
            if (dense) {
                for (int ti = 0; ti < T; ti++)
                    for (int tj = 0; tj < T; tj++)
                        h_A[(bi * T + ti) * N + bj * T + tj] = (float)rand() / RAND_MAX;
            }
        }
    }

    BCSR bcsr(h_A, tile_dense, M, N, TILING);
    free(tile_dense);

    float *d_A, *d_B, *d_naive, *d_sparse;
    cudaMalloc(&d_A,      bytes_A);
    cudaMalloc(&d_B,      bytes_B);
    cudaMalloc(&d_naive,  bytes_out);
    cudaMalloc(&d_sparse, bytes_out);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    matmul_naive(d_A, d_B, d_naive, M, N, K);
    spmm(bcsr, d_B, d_sparse, M, N, K);
    cudaDeviceSynchronize();

    float* h_naive  = (float*)malloc(bytes_out);
    float* h_sparse = (float*)malloc(bytes_out);
    cudaMemcpy(h_naive,  d_naive,  bytes_out, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sparse, d_sparse, bytes_out, cudaMemcpyDeviceToHost);

    float diff = max_abs_diff(h_naive, h_sparse, M * K);

    free(h_A); free(h_B); free(h_naive); free(h_sparse);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_naive); cudaFree(d_sparse);
    return diff;
}

// Run sddmm against the CPU reference. A is I x K, B is J x K (both row-major).
// D's sparsity pattern is generated at the requested density; the CPU and GPU
// outputs are compared element-wise on the .values strip.
static float run_sddmm_test(int I, int J, int K, float sparsity, int TILING) {
    int Tb_i = I / TILING;
    int Tb_j = J / TILING;

    size_t bytes_A = (size_t)I * K * sizeof(float);
    size_t bytes_B = (size_t)J * K * sizeof(float);

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    rand_fill(h_A, I * K);
    rand_fill(h_B, J * K);

    bool* tile_dense = (bool*)malloc((size_t)Tb_i * Tb_j * sizeof(bool));
    for (int bi = 0; bi < Tb_i; bi++)
        for (int bj = 0; bj < Tb_j; bj++)
            tile_dense[bi * Tb_j + bj] = ((float)rand() / RAND_MAX) >= sparsity;

    BCSR D_gpu(nullptr, tile_dense, I, J, TILING);
    BCSR D_cpu(nullptr, tile_dense, I, J, TILING);
    free(tile_dense);

    sddmm_cpu(h_A, h_B, D_cpu, K);

    float *d_A, *d_B;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    sddmm(d_A, d_B, D_gpu, K);
    cudaDeviceSynchronize();

    size_t nvals = (size_t)D_gpu.nnzb * TILING * TILING;
    float diff = max_abs_diff(D_gpu.values, D_cpu.values, (int)nvals);

    free(h_A); free(h_B);
    cudaFree(d_A); cudaFree(d_B);
    return diff;
}

struct TestCase { int M, N, K; };

int main() {
    srand(42);

    // M and K must be multiples of 32; N must be a multiple of 128 (TILE_K_STEP).
    TestCase cases[] = {
        {  32, 128,  32 },
        {  64, 128,  64 },
        {  64, 256,  64 },
        { 128, 256, 128 },
        { 256, 512, 128 },
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

    printf("\n--- sparse matmul (naive vs matmul_sparse) ---\n");
    printf("%-28s  %8s  %12s  %s\n", "Shape (MxNxK)", "Sparsity", "Max |diff|", "Result");
    printf("%-28s  %8s  %12s  %s\n", "----------------------------", "--------", "------------", "------");

    struct SparseCase { int M, N, K; float sparsity; };
    SparseCase scases[] = {
        {  32,  128,  32, 0.0f },   // fully dense — baseline
        {  32,  128,  32, 0.5f },   // 50% tiles sparse
        {  32,  128,  32, 0.9f },   // 90% tiles sparse
        {  64,  256,  64, 0.5f },   // larger matrix, half sparse
        { 128,  512, 128, 0.75f},   // stress, 75% sparse
    };

    int s_passed = 0, s_failed = 0;
    for (auto& sc : scases) {
        float diff = run_sparse_test(sc.M, sc.N, sc.K, sc.sparsity);
        bool ok = diff < kTol;
        ok ? s_passed++ : s_failed++;
        printf("%dx%dx%-18d  %7.0f%%  %12.2e  %s\n",
               sc.M, sc.N, sc.K, sc.sparsity * 100.f, diff, ok ? "PASS" : "FAIL");
    }

    printf("\n%d/%zu passed\n", s_passed, sizeof(scases) / sizeof(scases[0]));

    printf("\n--- sddmm (gpu vs cpu reference) ---\n");
    printf("%-22s  %4s  %8s  %12s  %s\n",
           "Shape (IxJxK)", "T", "Sparsity", "Max |diff|", "Result");
    printf("%-22s  %4s  %8s  %12s  %s\n",
           "----------------------", "----", "--------", "------------", "------");

    struct SddmmCase { int I, J, K; float sparsity; int TILING; };
    SddmmCase dcases[] = {
        // TILING=32 (Phase 1 baseline).
        {  64,  64,  32, 0.5f, 32 },
        { 128, 128, 128, 0.5f, 32 },
        { 128, 128, 128, 0.9f, 32 },
        // TILING=16 / 8 (Phase 2).
        { 128, 128, 128, 0.5f, 16 },
        { 128, 128, 128, 0.9f, 16 },
        { 128, 128, 128, 0.5f,  8 },
        { 256, 128, 256, 0.7f,  8 },
        // TILING=4 / 2 / 1 (Phase 3).
        { 128, 128, 128, 0.5f,  4 },
        { 128, 128, 128, 0.5f,  2 },
        {  64,  64,  64, 0.5f,  1 },
        {  64,  64,  64, 0.9f,  1 },
    };

    int d_passed = 0, d_failed = 0;
    for (auto& dc : dcases) {
        float diff = run_sddmm_test(dc.I, dc.J, dc.K, dc.sparsity, dc.TILING);
        bool ok = diff < kTol;
        ok ? d_passed++ : d_failed++;
        printf("%dx%dx%-14d  %4d  %7.0f%%  %12.2e  %s\n",
               dc.I, dc.J, dc.K, dc.TILING, dc.sparsity * 100.f,
               diff, ok ? "PASS" : "FAIL");
    }

    printf("\n%d/%zu passed\n", d_passed, sizeof(dcases) / sizeof(dcases[0]));

    return (failed + s_failed + d_failed) > 0 ? 1 : 0;
}
