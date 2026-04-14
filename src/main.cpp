#include "transformer_naive.cu"

int main() {
    int N = 1024; // Sequence length
    int d = 512;  // Embedding dimension

    // Allocate memory for input and output matrices
    float* q; // Query matrix
    float* k; // Key matrix
    float* v; // Value matrix
    float* output; // Output matrix

    // Load test data into q, k, v
    file* q_file = fopen("q.bin", "rb");
    file* k_file = fopen("k.bin", "rb");
    file* v_file = fopen("v.bin", "rb");
    q = (float*)malloc(N * d * sizeof(float));
    k = (float*)malloc(N * d * sizeof(float));
    v = (float*)malloc(N * d * sizeof(float));
    fread(q, sizeof(float), N * d, q_file);
    fread(k, sizeof(float), N * d, k_file);
    fread(v, sizeof(float), N * d, v_file);
    fclose(q_file);
    fclose(k_file);
    fclose(v_file);

    // Call the transformer forward pass
    TransformerNaive::forward(q, k, v, output, N, d);

    return 0;
}