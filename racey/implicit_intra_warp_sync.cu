#include <stdio.h>

#ifdef GLOBAL
__device__ char x[THREADS];
#endif

__global__ void racey_kernel() {
#ifdef SHARED
    __shared__ char x[THREADS];
#endif

#ifdef WW
    x[threadIdx.x] = threadIdx.x;
    x[THREADS - threadIdx.x - 1] = threadIdx.x;
#elif RW
    volatile char c = x[threadIdx.x];
    x[THREADS - threadIdx.x - 1] = threadIdx.x;
#endif

}

int main() {
    racey_kernel<<<BLOCKS,THREADS>>>();

    cudaDeviceReset();
    return 0;
}
