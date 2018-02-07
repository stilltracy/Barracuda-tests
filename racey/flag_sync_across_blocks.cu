#include <limits.h>
#include <stdio.h>

#ifdef NO_VOLATILE
__device__ int flag = 0;
#else
__device__ volatile int flag = 0;
#endif

__global__ void simple_kernel() {
    if (blockIdx.x == 0) {
        __threadfence();
#ifdef ATOMIC
        atomicExch((int*)&flag, 1);
#else
        flag = 1;
#endif
    }
    if (blockIdx.x == 1) {
#ifdef ATOMIC
        while (atomicMax((int*)&flag, 0) != 1);
#elif PTX_CA
        asm(
            "{                              \n\t"
            "   .reg .u32  t;               \n\t"
            "   .reg .pred p;               \n\t"
            "BB0_2:                         \n\t"
            "   ld.global.u32.ca t, [flag]; \n\t"
            "   setp.ne.s32 p, t, 1;        \n\t"
            "   @p bra  BB0_2;              \n\t"
            "}                              \n\t"
        );
#elif PTX_CG
        asm(
            "{                              \n\t"
            "   .reg .u32  t;               \n\t"
            "   .reg .pred p;               \n\t"
            "BB0_2:                         \n\t"
            "   ld.global.u32.cg t, [flag]; \n\t"
            "   setp.ne.s32 p, t, 1;        \n\t"
            "   @p bra  BB0_2;              \n\t"
            "}                              \n\t"
        );
#else
        while (flag != 1);
#endif
        __threadfence();
    }
}

int main() {
    simple_kernel<<<2,1>>>();

    cudaDeviceReset();
    return 0;
}
