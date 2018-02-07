#include <stdio.h>
#include <unistd.h>

struct IntSandwich {
  int beginning;
  int middle[1];
  int end;  
};

__global__ void access_offset_kernel() {
    struct IntSandwich *devMem = (struct IntSandwich*)malloc(sizeof(struct IntSandwich));
#ifdef R
    volatile int i = devMem->middle[OFFSET] = 42;
#elif W
    devMem->middle[OFFSET] = 42;
#endif
    free(devMem);
}

int main(int argc, char** argv) {
    cudaThreadSetLimit(cudaLimitMallocHeapSize, 2*sizeof(struct IntSandwich));
    access_offset_kernel<<<1,1>>>();

    cudaDeviceReset();
    return 0;
}
