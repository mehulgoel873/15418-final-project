# Dynamic Sparse Attention Exploration

The goal of this project is to explore ways to make a naive attention algorithm, comprised of the operations: Matmul, Softmax, Matmul, significantly faster. 

The nonlinearity of the softmax leads to significant parallel slowdowns on CUDA GPUs, so through implementations like Flash Attention (memory tiling), Sparse Attention (Only compute a subset of values), et cetera. 

Project Proposal: [pdf](project-proposal/proposal.pdf)

Midway Report: [pdf](milestone-report/midway.pdf)