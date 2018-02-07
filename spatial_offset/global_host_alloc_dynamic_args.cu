#include <stdio.h>
#include <unistd.h>

#define ALLOC_SIZE 1024

__global__ void access_offset_kernel(int *hostMem, int offset) {
#ifdef R
    if (offset >= 0)
        volatile int i = hostMem[(ALLOC_SIZE-1) + offset];
    else
        volatile int i = hostMem[offset];
#elif W
    if (offset >= 0)
        hostMem[(ALLOC_SIZE-1) + offset] = 42;
    else
        hostMem[offset] = 42;
#endif
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

    int *hostMem;
    cudaMalloc((void**)&hostMem, ALLOC_SIZE*sizeof(int));
    access_offset_kernel<<<1,1>>>(hostMem, offset);
    cudaFree(hostMem);

    cudaDeviceReset();
    return 0;
}
