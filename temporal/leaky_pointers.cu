#include <limits.h>
#include <stdio.h>

#ifdef SHARED_LEAK_ACROSS_BLOCKS
__device__ volatile int flag = 0;
#endif

#ifdef TO_GLOBAL_PTR
__device__ int *p;
#endif

__global__ void simple_kernel() {
#ifdef TO_SHARED_PTR
    __shared__ int *p;
#endif

#ifdef LOCAL_LEAK_WITHIN_WARP
    int i;
    if (threadIdx.x == 0)
        p = &i;
    if (threadIdx.x == 1)
        i = *p;
#elif SHARED_LEAK_ACROSS_BLOCKS
    __shared__ int i;

    if (blockIdx.x == 0) {
        p = &i;
        __threadfence();
        flag = 1;
    }
    if (blockIdx.x == 1) {
        while (flag != 1);
        __threadfence();
        i = *p;
    }
#endif
}

int main() {
#ifdef LOCAL_LEAK_WITHIN_WARP
    simple_kernel<<<1,2>>>();
#elif SHARED_LEAK_ACROSS_BLOCKS
    simple_kernel<<<2,1>>>();
#endif

    cudaDeviceReset();
    return 0;
}
