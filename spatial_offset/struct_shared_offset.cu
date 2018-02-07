#include <stdio.h>
#include <unistd.h>

struct IntSandwich {
  int beginning;
  int middle[1];
  int end;  
};

__global__ void access_offset_kernel() {
    __shared__ struct IntSandwich devMem;
    devMem.beginning = 0; devMem.middle[0] = 0; devMem.end = 0; 
#ifdef R
    volatile int i = devMem.middle[OFFSET];
#elif W
    devMem.middle[OFFSET] = 42;
    devMem.middle[OFFSET] = devMem.middle[OFFSET] * 2; // for unused warning
#endif
}

int main(int argc, char** argv) {
    access_offset_kernel<<<1,1>>>();

    cudaDeviceReset();
    return 0;
}
