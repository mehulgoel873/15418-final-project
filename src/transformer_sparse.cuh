#pragma once
#include <cuda_runtime.h>
#include "datastructures/bcsr.cuh"

class TransformerSparse {
public:
    // mask: a host-side flat boolean array of shape [N/granularity * N/granularity]
    void forward(float* q, float* k, float* v, float* output, int N, int d, const bool* mask, int granularity);
};
