// main.cu
/* Compile and run with:
nvcc main.cu -o run
./run
*/
#include <iostream>
#include <math.h>
#include <chrono>

#define NUM_BLOCKS 4
#define NUM_THREADS_PER_BLOCK 16

__global__ void add(int n, float* x, float* y) {
    // At each index, add x to y.
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    int N = 10000;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // Initialize our x and y arrays with some floats.
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Run the function on using the GPU.
    // <<NumBlocks, NumThreadsPerBlock>>
    add<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(N, x, y); // Notice the brackets.

    // Wait for GPU to finish before accessing on host
    // TODO: seems like it might be calling this before the kernels even start
    cudaDeviceSynchronize();

    auto t2 = std::chrono::high_resolution_clock::now();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    std::cout << "Running with <<" << NUM_BLOCKS << ", " << NUM_THREADS_PER_BLOCK << ">> took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
