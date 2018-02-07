#include <stdio.h>

#define ALLOC_SIZE 1024

int main() {
    int *hostAllocMem;

    cudaMalloc((void**)&hostAllocMem, ALLOC_SIZE*sizeof(int));
    cudaMemset(hostAllocMem, 0, (ALLOC_SIZE+1)*sizeof(int));
    cudaFree(hostAllocMem);
    
    cudaDeviceReset();
    return 0;
}
