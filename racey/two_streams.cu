#include <stdio.h>

__device__ char x = 0;

__global__ void racey_kernel() {
#ifdef WW
    x = threadIdx.x;
#elif RW
    volatile char c = x;
    x = c + 1;
#endif
}

int main() {
    cudaStream_t s1, s2;

    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    racey_kernel<<<1,1,0,s1>>>();

#ifdef DEVICE_SYNC
    cudaDeviceSynchronize();
#elif STREAM_SYNC
    cudaStreamSynchronize(s1);
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

    racey_kernel<<<1,1,0,s2>>>();

#if defined(EVENT_SYNC) || defined(STREAM_EVENT_SYNC)
    cudaEventDestroy(event);
#endif

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    cudaDeviceReset();
    return 0;
}
