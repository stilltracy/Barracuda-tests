#include <stdio.h>

__device__ char x = 0;


__global__ void child_kernel1() {
    x = threadIdx.x;
}

__global__ void child_kernel2() {
#ifdef WW
    x = threadIdx.x;
#elif RW
    volatile char c = x;
#endif
}

__global__ void parent_kernel() {
    cudaStream_t s1, s2;

    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

    child_kernel1<<<1,1,0,s1>>>();

#ifdef DEVICE_SYNC
    cudaDeviceSynchronize();
#elif EVENT_SYNC
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event);
    cudaEventSynchronize(event);
#elif STREAM_EVENT_SYNC
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, s1);
    cudaStreamWaitEvent(s2, event, 0);
#elif STREAM_EVENT_SYNC_NOOP
    cudaEvent_t event;
    cudaEventCreate(&event); 
    cudaStreamWaitEvent(s2, event, 0); // no-op without cudaEventRecord
#endif

    child_kernel2<<<1,1,0,s2>>>();

#if defined(EVENT_SYNC) || defined(STREAM_EVENT_SYNC)
    cudaEventDestroy(event);
#endif

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}


int main() {
    parent_kernel<<<1,1>>>();

    cudaDeviceReset();
    return 0;
}
