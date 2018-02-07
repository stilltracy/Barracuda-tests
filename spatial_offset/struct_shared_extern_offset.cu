#include <stdio.h>
#include <unistd.h>

struct IntSandwich {
  int beginning;
  int middle[1];
  int end;  
};

__global__ void access_offset_kernel() {
    extern __shared__ int devMem[];
    struct IntSandwich* ps = (struct IntSandwich*)devMem;
    ps->beginning = 0; ps->middle[0] = 0; ps->end = 0; 
#ifdef R
    volatile int i = ps->middle[OFFSET];
#elif W
    ps->middle[OFFSET] = 42;
#endif
}

int main(int argc, char** argv) {
    access_offset_kernel<<<1,1,sizeof(struct IntSandwich)>>>();

    cudaDeviceReset();
    return 0;
}
