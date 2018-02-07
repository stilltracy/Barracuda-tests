#include <stdio.h>

#ifdef GLOBAL
__device__ char x = 0;
#endif

__global__ void atomic_kernel() {
#ifdef SHARED
    __shared__ char x;
#endif

    // using char because only one byte in size
    if (THREADS == 2 || THREADS > 32 && threadIdx.x % 32 == 0)
        atomicInc((unsigned int*)&x, 64);
}

int main() {
    atomic_kernel<<<BLOCKS,THREADS>>>();

    cudaDeviceReset();
    return 0;
}
