#include <stdio.h>

#ifdef GLOBAL
__device__ char x = 0;
#endif

__global__ void racey_kernel() {
#ifdef SHARED
    __shared__ char x;
#endif

    if (threadIdx.x == 0)
        x = threadIdx.x;
    else
        volatile char c = x;
}

int main() {
    racey_kernel<<<BLOCKS,THREADS>>>();

    cudaDeviceReset();
    return 0;
}
