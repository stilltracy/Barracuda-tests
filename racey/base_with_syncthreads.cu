#include <stdio.h>

#ifdef GLOBAL
__device__ char x = 0;
#endif

__global__ void racey_kernel() {
#ifdef SHARED
    __shared__ char x;
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0)
#ifdef WW
        x = threadIdx.x + blockIdx.x;
#elif RW
        volatile char c = x;
#endif
    __syncthreads();
    if (threadIdx.x == 32 || blockIdx.x == 1)
        x = threadIdx.x;
}

int main() {
    racey_kernel<<<BLOCKS,THREADS>>>();

    cudaDeviceReset();
    return 0;
}
