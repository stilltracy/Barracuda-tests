#include <stdio.h>

#define ALLOC_SIZE 1024

#define ACCESS_DIR 1

__global__ void access_from_min_into_max_kernel(int *min, int *max) {
#ifdef R
    volatile int i = min[max-min];
#elif W
    min[max-min] = 42;
#endif
}

__global__ void access_from_max_into_min_kernel(int *min, int *max) {
#ifdef R
    volatile int i = max[-1*(max-(min+(ALLOC_SIZE-1)))];
#elif W
    max[-1*(max-(min+(ALLOC_SIZE-1)))] = 42;
#endif
}

int main(int argc, char** argv) {
    int *x, *y, *min, *max;
    cudaMalloc((void**)&x, ALLOC_SIZE*sizeof(int));
    cudaMalloc((void**)&y, ALLOC_SIZE*sizeof(int));

    min = (x < y) ? x : y;
    max = (x < y) ? y : x;

    if (ACCESS_DIR == 0)
        access_from_min_into_max_kernel<<<1,1>>>(min,max);
    else
        access_from_max_into_min_kernel<<<1,1>>>(min,max);

    cudaFree(x);
    cudaFree(y);
    
    cudaDeviceReset();
    return 0;
}
