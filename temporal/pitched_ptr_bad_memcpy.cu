#include <assert.h>
#include <limits.h>
#include <stdio.h>

#define ALLOC_SIZE 126
#define HTD_ROW_OFFSET 4
#define DTH_ROW_OFFSET 4

int main() {
    const unsigned int ALLOC_WIDTH_BYTES = ALLOC_SIZE*sizeof(int);

    cudaPitchedPtr devMem;
    cudaExtent extent = make_cudaExtent(ALLOC_WIDTH_BYTES, ALLOC_SIZE, 1);
    cudaMalloc3D(&devMem, extent);

    assert((devMem.pitch / 4) > ALLOC_SIZE + 1);

    int *hostMemPitchAligned = (int*)malloc(devMem.pitch*ALLOC_SIZE);
    memset(hostMemPitchAligned, 0, devMem.pitch);

    cudaMemcpy2D(devMem.ptr, devMem.pitch, 
                 hostMemPitchAligned, devMem.pitch, 
                 ALLOC_WIDTH_BYTES + (long)sizeof(int)*HTD_ROW_OFFSET,  
                 ALLOC_SIZE, 
                 cudaMemcpyHostToDevice);

    cudaMemcpy2D(hostMemPitchAligned, devMem.pitch, 
                 devMem.ptr, devMem.pitch, 
                 ALLOC_WIDTH_BYTES + (long)sizeof(int)*DTH_ROW_OFFSET, 
                 ALLOC_SIZE, 
                 cudaMemcpyDeviceToHost);

    cudaFree(devMem.ptr);
    free(hostMemPitchAligned);
    
    cudaDeviceReset();
    return 0;
}

