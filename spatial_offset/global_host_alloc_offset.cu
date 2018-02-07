#include <stdio.h>
#include <unistd.h>

#define ALLOC_SIZE 1024

__global__ void access_offset_kernel(int *hostMem) {
#ifdef R
    volatile int i = hostMem[OFFSET+ACCESS_DIR*(ALLOC_SIZE-1)];
#elif W
    hostMem[OFFSET+ACCESS_DIR*(ALLOC_SIZE-1)] = 42;
#endif
}

int main(int argc, char** argv) {
    int *hostMem;
    cudaMalloc((void**)&hostMem, ALLOC_SIZE*sizeof(int));
    access_offset_kernel<<<1,1>>>(hostMem);
    cudaFree(hostMem);

    cudaDeviceReset();
    return 0;
}
