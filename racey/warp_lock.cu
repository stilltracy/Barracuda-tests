#include <stdio.h>

#define LOCK_FREE 0
#define LOCK_HELD 1


/* A lock that can be called from just one thread within each warp. */
#ifdef NO_VOLATILE
__device__ unsigned theLock = LOCK_FREE;
#else
__device__ volatile unsigned theLock = LOCK_FREE;
#endif

__device__ void lock() {
		while (atomicCAS((unsigned int*)&theLock, LOCK_FREE, LOCK_HELD) != 0);
#ifndef NO_LOCK_FENCE
		__threadfence();
#endif
}

__device__ void unlock() {
#ifndef NO_UNLOCK_FENCE
		__threadfence();
#endif
		atomicExch((unsigned int*)&theLock, LOCK_FREE);
}


/* Test kernel */
#ifdef GLOBAL
__device__ char x = 0;
#endif

__global__ void lock_sync_kernel() {
#ifdef SHARED
    __shared__ char x;
#endif

    // using char because only one byte in size
    if (threadIdx.x % 32 == 0) {
        lock();
        x = threadIdx.x;
        unlock();
    }
}


int main() {
    lock_sync_kernel<<<BLOCKS,THREADS>>>();

    cudaDeviceReset();
    return 0;
}
