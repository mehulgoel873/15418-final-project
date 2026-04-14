#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <functional>

#include "transformer_naive.cu"
// #include "transformer_fast.cu", etc

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

int main(int argc, char** argv)
{
    int N      = 1024;
    int d      = 512;
    int iters  = 10;

    if (argc >= 3) { N = atoi(argv[1]); d = atoi(argv[2]); }
    if (argc >= 4) iters  = atoi(argv[3]);

    printf("N=%-6d  d=%-6d  iters=%d\n\n", N, d, iters);
    printf("%-28s  %10s\n", "Implementation", "ms/iter");
    printf("%-28s  %10s\n", "----------------------------", "----------");

    {
        TransformerNaive impl;
        float ms = benchmark(
            [&](float* q, float* k, float* v, float* out, int N, int d) {
                impl.forward(q, k, v, out, N, d);
            }, N, d, iters);
        printf("%-28s  %10.3f\n", "TransformerNaive", ms);
    }

    // Add new implementations here:
    // {
    //     TransformerFast impl;
    //     float ms = benchmark(
    //         [&](float* q, float* k, float* v, float* out, int N, int d) {
    //             impl.forward(q, k, v, out, N, d);
    //         }, N, d, warmup, iters);
    //     printf("%-28s  %10.3f\n", "TransformerFast", ms);
    // }

    return 0;
}
