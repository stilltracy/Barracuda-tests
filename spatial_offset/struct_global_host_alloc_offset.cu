#include <stdio.h>
#include <unistd.h>

struct IntSandwich {
  int beginning;
  int middle[1];
  int end;  
};

__global__ void access_offset_kernel(struct IntSandwich *hostMem) {
#ifdef R
    volatile int i = hostMem->middle[OFFSET] = 42;
#elif W
    hostMem->middle[OFFSET] = 42;
#endif
}

int main(int argc, char** argv) {
    struct IntSandwich *hostMem;
    cudaMalloc((void**)&hostMem, sizeof(struct IntSandwich));

    access_offset_kernel<<<1,1>>>(hostMem);
    cudaFree(hostMem);

    cudaDeviceReset();
    return 0;
}
