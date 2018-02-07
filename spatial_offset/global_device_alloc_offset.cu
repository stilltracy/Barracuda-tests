#include <stdio.h>
#include <unistd.h>

#define ALLOC_SIZE 1024

__global__ void access_offset_kernel() {
    int* devMem = (int*)malloc(ALLOC_SIZE*sizeof(int));
#ifdef R
    volatile int i = devMem[OFFSET+ACCESS_DIR*(ALLOC_SIZE-1)];
#elif W
    devMem[OFFSET+ACCESS_DIR*(ALLOC_SIZE-1)] = 42;
#endif
    free(devMem);
}

int main(int argc, char** argv) {
    cudaThreadSetLimit(cudaLimitMallocHeapSize, ALLOC_SIZE*4*sizeof(int));
    access_offset_kernel<<<1,1>>>();

    cudaDeviceReset();
    return 0;
}
