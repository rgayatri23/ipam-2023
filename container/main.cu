#include <iostream>
#include <chrono>
#include <sched.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std::chrono;

__global__ void kernel(int N, int *a, int *b, int *c)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < N)
        c[i] = a[i] * b[i];
}

int main(int argc, char **argv) {

    const int N = argc > 1 ? atoi(argv[1]) : 10;

    // host arrays
    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];

    // initialize inputs
    for(int i = 0; i < N; ++i)
    {
        a[i] = i+1;
        b[i] = i+2;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // do work
    const int threads = 32;
    const int blocks = N/threads + 1;
    kernel<<<blocks,threads>>> (N, d_a, d_b, d_c);
    cudaDeviceSynchronize();
    
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) 
        printf("c[%d] = %d \t - expected %d\n",i,c[i],(i+1)*(i+2));

    // cleanup
    delete[] a;
    delete[] b;
    delete[] c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
