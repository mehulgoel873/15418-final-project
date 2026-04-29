#pragma once
#include "datastructures/bcsr.cuh"

void matmul_tiled(float* A, float* B, float* output, int M, int N, int K);
void matmul_naive(float* A, float* B, float* output, int M, int N, int K);
void spmm(BCSR& A, float* B, float* output, int M, int N, int K);

// SDDMM: D = (A * B^T) hadamard I[mask(D)]. A is I x K, B is J x K (both
// row-major); D is BCSR with caller-provided sparsity pattern (I/T x J/T
// blocks). The kernel only writes the dense tiles in D.
void sddmm(const float* A, const float* B, BCSR& D, int K);
void sddmm_cpu(const float* A, const float* B, BCSR& D, int K);

// Multiply every dense value in D by s. Trivial elementwise pass over D.values.
void scale_bcsr_values(BCSR& D, float s);

// BCSR x BCSR -> BCSR. The caller must preconstruct `output` with the
// sparsity pattern returned by bcsr_matmul_mask(A, B); this kernel only
// fills values for tiles already marked dense in `output`.
void matmul_sparse_bcsr(BCSR& A, BCSR& B, BCSR& output);

