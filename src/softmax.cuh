#pragma once

/// Host launcher: sets up grid/block dims and calls softmax_blocked_kernel.
void softmax_tiled(float* input, float* output, int row_len, int num_rows);

/// Host launcher for the naive softmax kernel (16x16 blocks, no shared memory).
void softmax_naive(float* input, float* output, int row_len, int num_rows);