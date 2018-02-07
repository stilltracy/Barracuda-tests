#include <stdio.h>
#define SRC_SIZE 65536
#define DST_SIZE 65536
#define CPY_SIZE 8192

int main() {
    int *h_mem = (int*)malloc(SRC_SIZE*sizeof(int));

    int *d_mem;
    cudaMalloc((void**)&d_mem, DST_SIZE*sizeof(int));

    cudaMemcpy(d_mem, h_mem, CPY_SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h_mem, d_mem, CPY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_mem);
    free(h_mem);
    
    cudaDeviceReset();
    return 0;
}
