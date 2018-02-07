#include <stdio.h>
#include <unistd.h>

#define ALLOC_SIZE 1024

__global__ void access_offset_kernel(int offset) {
    int* devMem = (int*)malloc(ALLOC_SIZE*sizeof(int));
    
#ifdef R
    if (offset >= 0)
        volatile int i = devMem[(ALLOC_SIZE-1) + offset];
    else
        volatile int i = devMem[offset];
#elif W
    if (offset >= 0)
        devMem[(ALLOC_SIZE-1) + offset] = 42;
    else
        devMem[offset] = 42;
#endif
    
    free(devMem);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s -o <offset>\n", argv[0]);
        abort();
    }

    int offset = 0;
    int c;
    while ((c = getopt(argc, argv, "o:")) != -1) {
        switch(c) {
        case 'o':
            offset = atoi(optarg);
            break;
        default:
            fprintf(stderr, "Usage: %s -o <offset>\n", argv[0]);
            abort();
        }
    }

    cudaThreadSetLimit(cudaLimitMallocHeapSize, ALLOC_SIZE*4*sizeof(int));
    access_offset_kernel<<<1,1>>>(offset);

    cudaDeviceReset();
    return 0;
}
