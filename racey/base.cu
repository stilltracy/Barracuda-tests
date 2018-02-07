#include <stdio.h>

#ifdef GLOBAL
__device__ char x = 0;
#endif

__global__ void racey_kernel() {
#ifdef SHARED
    __shared__ char x;
#endif

#ifdef WW
    x = threadIdx.x;
    x = threadIdx.x + blockIdx.x;
#elif RW
    volatile char c = x;
    x = c + 1;
#endif
}

int main() {
    racey_kernel<<<BLOCKS,THREADS>>>();

    cudaDeviceReset();
    return 0;
}
