#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <functional>

// Global verbose flag. Set to true to enable timing prints; false to skip them.
// Use verbose_timing() = true/false to control from call sites.
inline bool& verbose_timing() { static bool v = false; return v; }

// Launch fn(), and — if verbose_timing() is set — measure GPU time with CUDA
// events and print the result. Always returns elapsed ms (0 if not verbose).
static inline float time_and_print(const char* label, std::function<void()> fn) {
    if (!verbose_timing()) { fn(); return 0.f; }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    fn();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("[%-36s]  %.3f ms\n", label, ms);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}
