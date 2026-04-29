#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <functional>

#include "transformer_naive.cuh"
#include "transformer_sparse.cuh"
#include "transformer_sparse_softmax.cuh"
#include "timing.cuh"

// x, mask, output, N, d, granularity
typedef std::function<void(float*, float*, float*, int, int, int)> ForwardFn;

static void rand_init_device(float* d_ptr, int n)
{
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX;
    cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

// Expand a (num_blocks x num_blocks) per-block decision array into a full
// (n x n) float mask. Each thread writes one mask element by looking up its
// owning block's decision: 1 -> -inf, 0 -> 1.0.
__global__ void expand_block_mask_kernel(float* mask, const unsigned char* block_is_inf,
                                         int num_blocks, int granularity, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    int br = row / granularity;
    int bc = col / granularity;
    mask[row * n + col] = block_is_inf[br * num_blocks + bc] ? -INFINITY : 0.0f;
}

// Fills `mask` (device memory, n x n) with a random block-structured additive
// mask. Each (granularity x granularity) tile is independently set to -inf
// with probability p, otherwise 0.0. Designed to be added to attention scores
// before softmax (standard masking). Assumes n % granularity == 0.
static void get_rand_mask(float* mask, float p, int granularity, int n) {
    int num_blocks = n / granularity;
    size_t nb2 = (size_t)num_blocks * num_blocks;

    unsigned char* h_blocks = (unsigned char*)malloc(nb2);
    for (size_t i = 0; i < nb2; i++) {
        h_blocks[i] = ((float)rand() / RAND_MAX) < p ? 1 : 0;
    }

    unsigned char* d_blocks;
    cudaMalloc(&d_blocks, nb2);
    cudaMemcpy(d_blocks, h_blocks, nb2, cudaMemcpyHostToDevice);
    free(h_blocks);

    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    expand_block_mask_kernel<<<grid, block>>>(mask, d_blocks, num_blocks, granularity, n);
    cudaDeviceSynchronize();
    cudaFree(d_blocks);
}


static void check_correctness(ForwardFn naive_fn, ForwardFn test_fn, int N, int d, float p, int sparse_granularity)
{
    float *d_x_naive, *d_x_test, *d_out_naive, *d_out_test, *d_mask_naive, *d_mask_test;
    size_t x_bytes = (size_t)N * d * sizeof(float);
    size_t mask_bytes = (size_t)N * N * sizeof(float);

    cudaMalloc(&d_x_naive,   x_bytes);
    cudaMalloc(&d_x_test,    x_bytes);
    cudaMalloc(&d_mask_naive, mask_bytes);
    cudaMalloc(&d_mask_test,  mask_bytes);
    cudaMalloc(&d_out_naive, x_bytes);
    cudaMalloc(&d_out_test,  x_bytes);

    // Initialize the naive input arrays
    rand_init_device(d_x_naive, N * d);
    get_rand_mask(d_mask_naive, p, sparse_granularity, N);

    // Copy to the test arrays to allow either function to modify inputs in-place
    cudaMemcpy(d_x_test, d_x_naive, x_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_mask_test, d_mask_naive, mask_bytes, cudaMemcpyDeviceToDevice);

    naive_fn(d_x_naive, d_mask_naive, d_out_naive, N, d, sparse_granularity);
    cudaDeviceSynchronize();
    test_fn(d_x_test, d_mask_test, d_out_test, N, d, sparse_granularity);
    cudaDeviceSynchronize();

    float *h_out_naive = (float*)malloc(x_bytes);
    float *h_out_test  = (float*)malloc(x_bytes);

    cudaMemcpy(h_out_naive, d_out_naive, x_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_test,  d_out_test,  x_bytes, cudaMemcpyDeviceToHost);

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < N * d; i++) {
        float err = fabs(h_out_naive[i] - h_out_test[i]);
        float rel = err / (fmaxf(fabs(h_out_naive[i]), fabs(h_out_test[i])) + 1e-5f);
        if (err > max_abs_err) max_abs_err = err;
        if (rel > max_rel_err) max_rel_err = rel;
    }

    if (max_rel_err < 1e-2f || max_abs_err < 1e-2f) {
        printf("Correctness check: PASS (max abs error: %e, max rel error: %e)\n", max_abs_err, max_rel_err);
    } else {
        printf("Correctness check: FAIL (max abs error: %e, max rel error: %e)\n", max_abs_err, max_rel_err);
    }

    free(h_out_naive);
    free(h_out_test);
    cudaFree(d_x_naive); cudaFree(d_mask_naive);
    cudaFree(d_x_test);  cudaFree(d_mask_test);
    cudaFree(d_out_naive); cudaFree(d_out_test);
}

static float benchmark(ForwardFn fn, int N, int d, float p, int sparse_granularity, int iters)
{

    float *x, *output, *mask;
    cudaMalloc(&x,      (size_t)N * d * sizeof(float));
    cudaMalloc(&mask,      (size_t)N * N * sizeof(float));
    cudaMalloc(&output, (size_t)N * d * sizeof(float));

    rand_init_device(x, N * d);
    get_rand_mask(mask, p, sparse_granularity, N);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) {
        verbose_timing() = (i == 0);
        fn(x, mask, output, N, d, sparse_granularity);
    }
    verbose_timing() = false;
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(x); cudaFree(mask); cudaFree(output);

    return ms / iters;
}

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [--impl <naive|sparse|sparse_softmax>] [--check] [--sparsity <0.0-1.0>] [--granularity <int>] [--seed <int>] [N d [iters]]\n"
            "  --impl        which transformer to run (default: naive)\n"
            "  --check       check correctness against naive implementation\n"
            "  --sparsity    percentage of attention tiles that are sparse (default: 0.5)\n"
            "  --granularity tile size for sparsity (default: 32)\n"
            "  --seed        RNG seed for reproducible inputs/masks (default: time(NULL))\n"
            "  N             sequence length        (default: 16384)\n"
            "  d             embedding dimension    (default: 768)\n"
            "  iters         benchmark iterations   (default: 10)\n",
            prog);
}

int main(int argc, char** argv)
{
    const char* impl = "naive";
    bool do_check = false;
    float sparsity = 0.5f;
    int granularity = 32;
    int N     = 16384;
    int d     = 768;
    int iters = 10;
    unsigned int seed = (unsigned int)time(NULL);
    bool seed_set = false;

    // Parse arguments (note: N d iters are positional args, must come after flags)
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--impl") == 0) {
            if (++i >= argc) { usage(argv[0]); return 1; }
            impl = argv[i];
        } else if (strcmp(argv[i], "--check") == 0) {
            do_check = true;
        } else if (strcmp(argv[i], "--sparsity") == 0) {
            if (++i >= argc) { usage(argv[0]); return 1; }
            sparsity = atof(argv[i]);
        } else if (strcmp(argv[i], "--granularity") == 0) {
            if (++i >= argc) { usage(argv[0]); return 1; }
            granularity = atoi(argv[i]);
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (++i >= argc) { usage(argv[0]); return 1; }
            seed = (unsigned int)strtoul(argv[i], NULL, 10);
            seed_set = true;
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

    srand(seed);
    printf("Using RNG seed: %u%s\n", seed, seed_set ? "" : " (auto)");

    if (do_check && strcmp(impl, "naive") != 0) {
        printf("Running correctness check against 'naive'...\n");
        // Re-seed before each constructor so both transformers consume the
        // same RNG stream and end up with identical W_q/W_k/W_v weights.
        srand(seed);
        TransformerNaive naive_t(d);
        auto naive_fn = [&](float* x, float* mask, float* out, int N, int d, int g) {
            naive_t.forward(x, mask, out, N, d, g);
        };

        if (strcmp(impl, "sparse") == 0) {
            srand(seed);
            TransformerSparse t(d);
            auto test_fn = [&](float* x, float* mask, float* out, int N, int d, int g) {
                t.forward(x, mask, out, N, d, g);
            };
            check_correctness(naive_fn, test_fn, N, d, sparsity, granularity);
        } else if (strcmp(impl, "sparse_softmax") == 0) {
            srand(seed);
            TransformerSparseSoftmax t(d);
            auto test_fn = [&](float* x, float* mask, float* out, int N, int d, int g) {
                t.forward(x, mask, out, N, d, g);
            };
            check_correctness(naive_fn, test_fn, N, d, sparsity, granularity);
        }
        printf("\n");
    } else if (do_check) {
        printf("Skipping correctness check: '--impl naive' selected.\n\n");
    }

    const char* display_name = nullptr;
    float ms = 0.0f;

    // Each lambda matches ForwardFn: (x, mask, output, N, d, granularity).
    if (strcmp(impl, "naive") == 0) {
        TransformerNaive t(d);
        display_name = "TransformerNaive";
        ms = benchmark([&](float* x, float* mask, float* out, int N, int d, int g) {
            t.forward(x, mask, out, N, d, g);
        }, N, d, sparsity, granularity, iters);
    } else if (strcmp(impl, "sparse") == 0) {
        TransformerSparse t(d);
        display_name = "Transformer Sparse";
        ms = benchmark([&](float* x, float* mask, float* out, int N, int d, int g) {
            t.forward(x, mask, out, N, d, g);
        }, N, d, sparsity, granularity, iters);
    } else if (strcmp(impl, "sparse_softmax") == 0) {
        TransformerSparseSoftmax t(d);
        display_name = "Transformer Sparse Softmax";
        ms = benchmark([&](float* x, float* mask, float* out, int N, int d, int g) {
            t.forward(x, mask, out, N, d, g);
        }, N, d, sparsity, granularity, iters);
    } else {
        fprintf(stderr, "Unknown impl '%s'. Choose: naive, sparse\n", impl);
        usage(argv[0]); return 1;
    }

    printf("N=%-6d  d=%-6d  iters=%-6d", N, d, iters);
    if (strcmp(impl, "sparse") == 0) {
        printf("  sparsity=%-6.2f%%  granularity=%-6d\n\n", sparsity * 100.0f, granularity);
    }
    printf("%-28s  %10s\n", "Implementation", "ms/iter");
    printf("%-28s  %10s\n", "----------------------------", "----------");
    printf("%-28s  %10.3f\n", display_name, ms);

    return 0;
}
