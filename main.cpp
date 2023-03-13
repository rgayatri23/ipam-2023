#include <iostream>
#include <chrono>

using namespace std::chrono;

void random_ints(int *a, int N) {
    int i;
    for (i = 0; i < N; ++i) a[i] = rand();
}

int main(int argc, char **argv) {

    const int N = argc > 1 ? atoi(argv[1]) : 10;

    // host arrays
    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];

    // initialize inputs
    random_ints(a, N);
    random_ints(b, N);

    time_point<std::chrono::system_clock> start, end;

    start = system_clock::now();
    // do work
#if _OPENMP
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

    return 0;
}
