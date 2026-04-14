#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <functional>

#include "transformer_naive.cu"
#include "transformer_tiled_matmul.cu"

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
    for (int i = 0; i < iters; i++)
        fn(q, k, v, output, N, d);
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
            "Usage: %s [--impl <naive|tiled>] [N d [iters]]\n"
            "  --impl  which transformer to run (default: naive)\n"
            "  N       sequence length        (default: 4096)\n"
            "  d       embedding dimension    (default: 4096)\n"
            "  iters   benchmark iterations   (default: 10)\n",
            prog);
}

int main(int argc, char** argv)
{
    const char* impl = "naive";
    int N     = 4096;
    int d     = 4096;
    int iters = 10;

    // Parse --impl <name> first, then remaining positional args N d iters.
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--impl") == 0) {
            if (++i >= argc) { usage(argv[0]); return 1; }
            impl = argv[i];
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

    printf("N=%-6d  d=%-6d  iters=%d\n\n", N, d, iters);
    printf("%-28s  %10s\n", "Implementation", "ms/iter");
    printf("%-28s  %10s\n", "----------------------------", "----------");

    if (strcmp(impl, "naive") == 0) {
        TransformerNaive t;
        float ms = benchmark([&](float* q, float* k, float* v, float* out, int N, int d) {
            t.forward(q, k, v, out, N, d);
        }, N, d, iters);
        printf("%-28s  %10.3f\n", "TransformerNaive", ms);
    } else if (strcmp(impl, "tiled") == 0) {
        TransformerTiledMatmul t;
        float ms = benchmark([&](float* q, float* k, float* v, float* out, int N, int d) {
            t.forward(q, k, v, out, N, d);
        }, N, d, iters);
        printf("%-28s  %10.3f\n", "TransformerTiledMatmul", ms);
    } else {
        fprintf(stderr, "Unknown impl '%s'. Choose: naive, tiled\n", impl);
        usage(argv[0]); return 1;
    }

    return 0;
}
