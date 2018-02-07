#include <limits.h>
#include <stdio.h>

int main() {
    int *hostMem;

    cudaMalloc((void**)&hostMem, ULONG_MAX);
    cudaFree(hostMem);
    
    cudaDeviceReset();
    return 0;
}
