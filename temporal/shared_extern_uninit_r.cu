#include <limits.h>
#include <stdio.h>

#define ALLOC_SIZE 1024

__global__ void simple_kernel() {
    extern __shared__ int devMem[];
    int i = devMem[0];
    i = i*i; // for unreferenced warning
}

int main() {
    simple_kernel<<<1,1,ALLOC_SIZE*sizeof(int)>>>();

    cudaDeviceReset();
    return 0;
}
