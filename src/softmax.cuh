#pragma once
#include "datastructures/bcsr.cuh"

// Set to true to use dynamic scheduling (atomic counter), false for static scheduling (weighted partitions)
constexpr bool SOFTMAX_BCSR_USE_DYNAMIC = false;

void softmax_tiled(float* input, float* output, int row_len, int num_rows);
void softmax_naive(float* input, float* output, int row_len, int num_rows);

// Softmax over a BCSR input, producing a dense output. The sparsity pattern
// is not exploited: sparse tiles are treated as zeros and each thread reads
// the BCSR lookup table for every element in its row.
void softmax_bcsr(BCSR& input, float* output);

// Softmax over a BCSR input, producing a BCSR output with the SAME sparsity
// pattern as the input. Sparse tiles in the input are treated as -INFINITY
// (contributing 0 after exp), so the output inherits the same zeroed-out tiles.
// The caller must preconstruct `output` with the same tile_dense mask as `input`.
void softmax_bcsr_bcsr(BCSR& input, BCSR& output);