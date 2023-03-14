#include <iostream>
#include <chrono>
#include <mpi.h>
#include <sched.h>

#if CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

using namespace std::chrono;

void random_ints(int *a, int N) {
    int i;
    for (i = 0; i < N; ++i) a[i] = rand();
}

#if CUDA
__global__ void kernel(int N, int *a, int *b, int *c)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x; 
    if(index < N)
        c[index] = a[index] + b[index];
}
#endif

int main(int argc, char **argv) {
    int myid, namelen, world_size;
    char myname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(myname, &namelen);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char* lrank = getenv("SLURM_PROCID");

    printf("Lrank from MPI = %s", lrank);

    char my_gpu[15];

    fprintf(stdout,
            "Hello Rahul from processor %s, rank = %d out of %d processors"
            "\n",
            myname, myid, world_size);

    // synchronize so the loop guarantees to prints all the information of one
    // rank before progressing.
    MPI_Barrier(MPI_COMM_WORLD);

    const int N = argc > 1 ? atoi(argv[1]) : 10;

    // host arrays
    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];

    // initialize inputs
    random_ints(a, N);
    random_ints(b, N);

#if CUDA
    int *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc(&d_a, N * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_b, N * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_c, N * sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
#endif

    time_point<std::chrono::system_clock> start, end;

    start = system_clock::now();
    // do work
#if CUDA
    const int threads = 32;
    const int blocks = N/threads + 1;
    kernel<<<blocks,threads>>> (N, d_a, d_b, d_c);
#else
#if defined(_OPENMP)
#if OPENMP_TARGET
#pragma omp target teams distribute parallel for \
    map(to: a[:N], b[:N]) map(from: c[:N])
#else
#pragma omp parallel for
#endif
#elif _OPENACC
#pragma acc parallel loop gang vector \
    copyin(a[:N], b[:N]) copyout(c[:N])
#endif
#endif
    for (int i = 0; i < N; i++) 
        c[i] = a[i] + b[i];

    end = system_clock::now();
    duration<double> elapsed = end - start;

    printf("Time to solution = %f [secs]\n", elapsed.count());


#ifdef DEBUG
    for (int i = 0; i < N; i++) 
        printf("c[%d] = %d\n",i,c[i]);
#endif

    // cleanup
    delete[] a;
    delete[] b;
    delete[] c;

#if CUDA
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
#endif

    fprintf(stdout,
            "\n****************************************************************"
            "******************** \n");
    MPI_Finalize();

    return 0;
}
