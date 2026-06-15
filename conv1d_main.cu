#include <cuda_runtime.h>
#include <stdio.h>

#include "conv1d.cu"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main() {
    float h_input[]  = {1, 2, 3, 4, 5};
    float h_kernel[] = {1, 0, -1};
    int input_size   = 5;
    int kernel_size  = 3;
    int output_size  = input_size - kernel_size + 1;

    float h_output[6] = {0};

    float *d_input, *d_kernel, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  input_size  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input,  h_input,  input_size  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_input, d_kernel, d_output, input_size, kernel_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Input:  ");
    for (int i = 0; i < input_size; i++) printf("%.1f ", h_input[i]);

    printf("\nKernel: ");
    for (int i = 0; i < kernel_size; i++) printf("%.1f ", h_kernel[i]);

    printf("\nOutput: ");
    for (int i = 0; i < output_size; i++) printf("%.1f ", h_output[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return 0;
}
