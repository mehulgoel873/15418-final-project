# Blocked Dynamic Sparse Attention
<sub>Saransh Agrawal, Mehul Goel</sub>

The goal of our project is to explore techniques to accelerate the sparse attention mechanism: the Matmul-Softmax-Matmul pipeline, by developing highly parallelized and efficient sparsity-aware kernels in CUDA.

### Final Report: [pdf](final-report/main.pdf)

As Large Language Models scale to massive context lengths, the standard attention mechanism is a severe computational bottleneck. While exploiting sparsity reduces training and inference costs, efficiently parallelizing these operations on GPUs introduces load imbalance, warp divergence, and uncoalesced accesses that perform poorly under the SIMT execution model. 

To overcome this, we enforce block-level sparsity using a Blocked Compressed Sparse Row (BCSR) format to preserve contiguous memory accesses and cache locality. We implemented three core CUDA kernels: Sampled Dense-Dense Matrix Multiplication (SDDMM), Sparse Softmax, and Sparse-Dense Matrix Multiplication (SpMM), utilizing dynamic warp-level active-coordinate gathering, static partitioned scheduling, and row-interleaved memory layouts to hide latency and maintain SM occupancy. 

Our evaluations on an RTX 6000 demonstrate that at large context lengths (N≥16384) and coarse granularities (G≥8), our sparse implementation approaches theoretical maximum speedups (1/(1-p)) with SDDMM eliminating wasted arithmetic, Sparse Softmax achieving granularity-invariant speedups via coalesced reductions, and SpMM scaling cleanly by maximizing data reuse along aligned block reads.


### Project Proposal: [pdf](project-proposal/proposal.pdf)

### Midway Report: [pdf](milestone-report/midway.pdf)