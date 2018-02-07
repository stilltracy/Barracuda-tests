#include <limits.h>
#include <stdio.h>

struct st {
  int* p;
};
__device__ struct st s;

__global__ void child_kernel(int* p) {}

__global__ void simple_kernel() {
#ifdef LEAK_SHARED
    __shared__ int i;
#elif LEAK_LOCAL
    int i;
#endif

#ifdef DIRECT
    // leak local/shared address directly to the child kernel launch
    child_kernel<<<1,1>>>(&i);
#elif INDIRECT
    // leak local/shared address indirectly via a global struct. Arguably, the leak occurs at the write to s.p
    s.p = &i;
    child_kernel<<<1,1>>>(s.p);
#endif

}

int main() {
    simple_kernel<<<1,1>>>();

    cudaDeviceReset();
    return 0;
}
