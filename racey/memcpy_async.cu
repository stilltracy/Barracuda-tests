#include <stdio.h>

#define ALLOC_SIZE 1024

__global__ void racey_kernel(char* d_mem) {
#ifdef WW
    d_mem[0] = threadIdx.x;
#elif RW
    volatile char c = d_mem[0];
#endif
}

int main() {
    char *h_mem = (char*)malloc(ALLOC_SIZE*sizeof(char));
    memset(h_mem, 0, ALLOC_SIZE*sizeof(char));

    char* d_mem;
    cudaMalloc((void**)&d_mem, ALLOC_SIZE*sizeof(char));
    cudaMemset(d_mem, 0, ALLOC_SIZE*sizeof(char));

#ifdef NON_DEFAULT_STREAM
    cudaStream_t s1;
    cudaStreamCreate(&s1);
    cudaMemcpyAsync(d_mem, h_mem, ALLOC_SIZE*sizeof(char), cudaMemcpyHostToDevice, s1);
    racey_kernel<<<1,1,0,s1>>>(d_mem);
    cudaStreamDestroy(s1);
#elif TWO_STREAM
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaMemcpyAsync(d_mem, h_mem, ALLOC_SIZE*sizeof(char), cudaMemcpyHostToDevice, s1);
    racey_kernel<<<1,1,0,s2>>>(d_mem);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
#elif TWO_STREAM_SYNC
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaMemcpyAsync(d_mem, h_mem, ALLOC_SIZE*sizeof(char), cudaMemcpyHostToDevice, s1);
    cudaStreamSynchronize(s1);
    racey_kernel<<<1,1,0,s2>>>(d_mem);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
#else
    cudaMemcpyAsync(d_mem, h_mem, ALLOC_SIZE*sizeof(char), cudaMemcpyHostToDevice);
    racey_kernel<<<1,1>>>(d_mem);
#endif

    cudaFree(d_mem);
    free(h_mem);

    cudaDeviceReset();
    return 0;
}
