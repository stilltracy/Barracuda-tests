#include <stdio.h>

__device__ char x = 0;

__global__ void racey_kernel(bool isParent) {
    if (isParent) {
        racey_kernel<<<1,1>>>(false);
    }
#ifdef WW
    x = threadIdx.x;
#elif RW
    volatile char c = x;
    x = c + 1;
#endif
}

__global__ void dynamic_kernel_parent_leads(bool isParent) {
    if (isParent) {
        x = threadIdx.x;
        racey_kernel<<<1,1>>>(false);
    } else {
#ifdef WW
        x = threadIdx.x;
#elif RW
        volatile char c = x;
#endif
    }
}

__global__ void dynamic_kernel_parent_waits(bool isParent) {
    if (isParent) {
        racey_kernel<<<1,1>>>(false);
        cudaDeviceSynchronize();
#ifdef WW
        x = threadIdx.x;
#elif RW
        volatile char c = x;
#endif
    } else {
        x = threadIdx.x;
    }
}


int main() {
#ifdef PARENT_LEADS
    dynamic_kernel_parent_leads<<<1,1>>>(true);
#elif PARENT_WAITS
    dynamic_kernel_parent_waits<<<1,1>>>(true);
#else
    racey_kernel<<<1,1>>>(true);
#endif

    cudaDeviceReset();
    return 0;
}
