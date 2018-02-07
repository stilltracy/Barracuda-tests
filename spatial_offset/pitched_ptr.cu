#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#define ALLOC_SIZE 126
#define ACCESS_MODE 0

__global__ void pitched_offset_negative_one(cudaPitchedPtr devMem) {
    int *d_p = (int*)devMem.ptr;
#ifdef R
    volatile int i = d_p[-1];
#elif W
    d_p[-1] = 42;
#endif
}

__global__ void pitched_offset_pitch(cudaPitchedPtr devMem) {
    int *d_p = &((int*)devMem.ptr)[ALLOC_SIZE-1]; 
#ifdef R
    volatile int i = d_p[1]; // forward into unused pitch
#elif W
    d_p[1] = 42; // forward into unused pitch
#endif
}

__global__ void pitched_offset_pitch_minus_one(cudaPitchedPtr devMem) {
    int *d_p = &((int*)devMem.ptr)[devMem.pitch/4]; 
#ifdef R
    volatile int i = d_p[-1]; // backward into unused pitch
#elif W
    d_p[-1] = 42; // backward into unused pitch
#endif
}

int main() {
    const int ALLOC_WIDTH_BYTES = ALLOC_SIZE*sizeof(int);
    const int ALLOC_TOTAL_BYTES = ALLOC_SIZE*ALLOC_WIDTH_BYTES;

    int *hostMem = (int*)malloc(ALLOC_TOTAL_BYTES);
    memset(hostMem, 0, ALLOC_TOTAL_BYTES);

    cudaPitchedPtr devMem;
    cudaExtent extent = make_cudaExtent(ALLOC_WIDTH_BYTES, ALLOC_SIZE, 1);
    cudaMalloc3D(&devMem, extent);

    assert((devMem.pitch / 4) > ALLOC_SIZE + 1);

    cudaMemcpy2D(devMem.ptr, devMem.pitch, hostMem, ALLOC_WIDTH_BYTES, 
                 ALLOC_WIDTH_BYTES, ALLOC_SIZE, cudaMemcpyHostToDevice);

    if (ACCESS_MODE == -1)
        pitched_offset_negative_one<<<1,1>>>(devMem);
    else if (ACCESS_MODE == 0)
        pitched_offset_pitch<<<1,1>>>(devMem);
    else
        pitched_offset_pitch_minus_one<<<1,1>>>(devMem);

    cudaFree(devMem.ptr);
    free(hostMem);
    
    cudaDeviceReset();
    return 0;
}
