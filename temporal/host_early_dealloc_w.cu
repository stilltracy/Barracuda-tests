#include <stdio.h>

#define ALLOC_SIZE 1024

__global__ void simple_kernel(int *hostMem) {
    hostMem[0] = 42;
}

int main() {
    int *hostMem;

    cudaMalloc((void**)&hostMem, ALLOC_SIZE*sizeof(int));
    cudaFree(hostMem);
    
    simple_kernel<<<1, 1>>>(hostMem);

    cudaDeviceReset();
    return 0;
}
