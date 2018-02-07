#include <stdio.h>

__device__ char x = 0;

__global__ void racey_kernel() {
#ifdef WW
    x = threadIdx.x;
#elif RW
    volatile char c = x;
    x = c + 1;
#endif
}

int main() {
    // sanity check test, would have been too messy to shoehorn into two_streams.cu
    racey_kernel<<<1,1>>>();
    racey_kernel<<<1,1>>>();

    cudaDeviceReset();
    return 0;
}
