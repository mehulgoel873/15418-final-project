#pragma once
#include "datastructures/bcsr.cuh"

void matmul_tiled(float* A, float* B, float* output, int M, int N, int K);
void matmul_naive(float* A, float* B, float* output, int M, int N, int K);
void matmul_sparse(BCSR& A, float* B, float* output, int M, int N, int K);
