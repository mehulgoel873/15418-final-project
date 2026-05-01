# Dynamic Sparse Attention Exploration

As modern LLMs scale to context lengths in the millions of tokens, the standard attention mechanism becomes a major computational bottleneck. The attention matrix is approximately sparse: most token interactions produce small outputs, and prior work has shown transformers can hold above 95% accuracy at 90% sparsity. Exploiting that sparsity on a GPU is non-trivial. Irregular skips create load imbalance, warp divergence, and uncoalesced memory accesses that fight the hardware's SIMT execution model.

We enforce sparsity at block granularity *G* using the Blocked Compressed Sparse Row (BCSR) format, which preserves contiguous memory accesses, vectorization, and cache locality within each block. On top of BCSR we implemented three CUDA kernels (**SDDMM**, **Sparse Softmax**, and **SpMM**) using warp-level active-coordinate gathering, partitioned scheduling, and row-interleaved memory layouts to hide latency and keep the SMs busy.

End-to-end on an RTX 6000, our sparse implementation approaches the theoretical 1/(1−*p*) speedup at large context lengths (*N* ≥ 16384) and coarse granularities (*G* ≥ 8). Per kernel:

- **SDDMM** eliminates wasted arithmetic but is shared-memory-bandwidth-bound at scale.
- **Sparse Softmax** achieves granularity-invariant speedups through coalesced reductions, especially at long contexts.
- **SpMM** scales cleanly by maximizing data reuse along contiguous block reads.


Project Proposal: [pdf](project-proposal/proposal.pdf)

Midway Report: [pdf](milestone-report/midway.pdf)

Final Report: [pdf](final-report/main.pdf)