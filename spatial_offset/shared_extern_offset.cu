#include <stdio.h>
#include <unistd.h>

#define ALLOC_SIZE 1024

__global__ void access_offset_kernel() {
    extern __shared__ int devMem[];
    devMem[0] = 0; devMem[1] = devMem[0]; // for init/unused warnings
#ifdef R
    volatile int i = devMem[OFFSET+ACCESS_DIR*(ALLOC_SIZE-1)];
#elif W
    devMem[OFFSET+ACCESS_DIR*(ALLOC_SIZE-1)] = 42;
#endif
}

int main(int argc, char** argv) {
    access_offset_kernel<<<1,1,ALLOC_SIZE*sizeof(int)>>>();

    cudaDeviceReset();
    return 0;
}
