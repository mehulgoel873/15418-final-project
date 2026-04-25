#pragma once
#include "datastructures/bcsr.cuh"

void matmul_tiled(float* A, float* B, float* output, int M, int N, int K);
void matmul_naive(float* A, float* B, float* output, int M, int N, int K);
void matmul_sparse(BCSR& A, float* B, float* output, int M, int N, int K);

// BCSR x BCSR -> BCSR. The caller must preconstruct `output` with the
// sparsity pattern returned by bcsr_matmul_mask(A, B); this kernel only
// fills values for tiles already marked dense in `output`.
void matmul_sparse_bcsr(BCSR& A, BCSR& B, BCSR& output);

