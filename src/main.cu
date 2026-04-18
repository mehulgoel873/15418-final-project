#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <functional>

#include "transformer_naive.cuh"
#include "transformer_tiled_matmul.cuh"
#include "transformer_tiled.cuh"
#include "timing.cuh"

// q, k, v, output, N, d
typedef std::function<void(float*, float*, float*, float*, int, int)> ForwardFn;

static void rand_init_device(float* d_ptr, int n)
{
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX;
    cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

// Can also include init from file code here

static void check_correctness(ForwardFn naive_fn, ForwardFn test_fn, int N, int d)
{
    float *d_q, *d_k, *d_v, *d_out_naive, *d_out_test;
    size_t bytes = (size_t)N * d * sizeof(float);
    cudaMalloc(&d_q, bytes);
    cudaMalloc(&d_k, bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_out_naive, bytes);
    cudaMalloc(&d_out_test,  bytes);

    rand_init_device(d_q, N * d);
    rand_init_device(d_k, N * d);
    rand_init_device(d_v, N * d);

    naive_fn(d_q, d_k, d_v, d_out_naive, N, d);
    cudaDeviceSynchronize();
    test_fn(d_q, d_k, d_v, d_out_test, N, d);
    cudaDeviceSynchronize();

    float *h_out_naive = (float*)malloc(bytes);
    float *h_out_test  = (float*)malloc(bytes);

    cudaMemcpy(h_out_naive, d_out_naive, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_test,  d_out_test,  bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < N * d; i++) {
        float err = fabs(h_out_naive[i] - h_out_test[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err < 1e-3f) {
        printf("Correctness check: PASS (max error: %e)\n", max_err);
    } else {
        printf("Correctness check: FAIL (max error: %e)\n", max_err);
    }

    free(h_out_naive);
    free(h_out_test);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_out_naive); cudaFree(d_out_test);
}

static float benchmark(ForwardFn fn, int N, int d, int iters)
{
    float *q, *k, *v, *output;
    cudaMalloc(&q,      (size_t)N * d * sizeof(float));
    cudaMalloc(&k,      (size_t)N * d * sizeof(float));
    cudaMalloc(&v,      (size_t)N * d * sizeof(float));
    cudaMalloc(&output, (size_t)N * d * sizeof(float));

    rand_init_device(q, N * d);
    rand_init_device(k, N * d);
    rand_init_device(v, N * d);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) {
        verbose_timing() = (i == 0);
        fn(q, k, v, output, N, d);
    }
    verbose_timing() = false;
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(output);

    return ms / iters;
}

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [--impl <naive|tiled>] [--check] [N d [iters]]\n"
            "  --impl  which transformer to run (default: naive)\n"
            "  --check check correctness against naive implementation\n"
            "  N       sequence length        (default: 16384)\n"
            "  d       embedding dimension    (default: 8192)\n"
            "  iters   benchmark iterations   (default: 10)\n",
            prog);
}

int main(int argc, char** argv)
{
    const char* impl = "naive";
    bool do_check = false;
    int N     = 16384;
    int d     = 8192;
    int iters = 10;

    // Parse --impl <name> first, then remaining positional args N d iters.
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--impl") == 0) {
            if (++i >= argc) { usage(argv[0]); return 1; }
            impl = argv[i];
        } else if (strcmp(argv[i], "--check") == 0) {
            do_check = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else {
            switch (pos++) {
                case 0: N     = atoi(argv[i]); break;
                case 1: d     = atoi(argv[i]); break;
                case 2: iters = atoi(argv[i]); break;
            }
        }
    }

    if (do_check && strcmp(impl, "naive") != 0) {
        printf("Running correctness check against 'naive'...\n");
        TransformerNaive naive_t;
        auto naive_fn = [&](float* q, float* k, float* v, float* out, int N, int d) {
            naive_t.forward(q, k, v, out, N, d);
        };

        if (strcmp(impl, "tiled_matmul") == 0) {
            TransformerTiledMatmul t;
            auto test_fn = [&](float* q, float* k, float* v, float* out, int N, int d) {
                t.forward(q, k, v, out, N, d);
            };
            check_correctness(naive_fn, test_fn, N, d);
        } else if (strcmp(impl, "tiled") == 0) {
            TransformerTiled t;
            auto test_fn = [&](float* q, float* k, float* v, float* out, int N, int d) {
                t.forward(q, k, v, out, N, d);
            };
            check_correctness(naive_fn, test_fn, N, d);
        }
        printf("\n");
    } else if (do_check) {
        printf("Skipping correctness check: '--impl naive' selected.\n\n");
    }

    if (strcmp(impl, "naive") == 0) {
        TransformerNaive t;
        float ms = benchmark([&](float* q, float* k, float* v, float* out, int N, int d) {
            t.forward(q, k, v, out, N, d);
        }, N, d, iters);

        printf("N=%-6d  d=%-6d  iters=%d\n\n", N, d, iters);
        printf("%-28s  %10s\n", "Implementation", "ms/iter");
        printf("%-28s  %10s\n", "----------------------------", "----------");
        printf("%-28s  %10.3f\n", "TransformerNaive", ms);
    } else if (strcmp(impl, "tiled_matmul") == 0) {
        TransformerTiledMatmul t;
        float ms = benchmark([&](float* q, float* k, float* v, float* out, int N, int d) {
            t.forward(q, k, v, out, N, d);
        }, N, d, iters);

        printf("N=%-6d  d=%-6d  iters=%d\n\n", N, d, iters);
        printf("%-28s  %10s\n", "Implementation", "ms/iter");
        printf("%-28s  %10s\n", "----------------------------", "----------");
        printf("%-28s  %10.3f\n", "Transformer Tiled Matmul", ms);
    } else if (strcmp(impl, "tiled") == 0) {
        TransformerTiled t;
        float ms = benchmark([&](float* q, float* k, float* v, float* out, int N, int d) {
            t.forward(q, k, v, out, N, d);
        }, N, d, iters);

        printf("N=%-6d  d=%-6d  iters=%d\n\n", N, d, iters);
        printf("%-28s  %10s\n", "Implementation", "ms/iter");
        printf("%-28s  %10s\n", "----------------------------", "----------");
        printf("%-28s  %10.3f\n", "Transformer Tiled", ms);
    } else {
        fprintf(stderr, "Unknown impl '%s'. Choose: naive, tiled_matmul, tiled\n", impl);
        usage(argv[0]); return 1;
    }

    return 0;
}
