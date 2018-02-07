#include <stdio.h>
#include <unistd.h>

#define ALLOC_SIZE 1024

__global__ void access_offset_kernel(int offset) {
    extern __shared__ int devMem[];
    devMem[0] = 0; devMem[1] = devMem[0]; // for init/unused warnings

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
    
    access_offset_kernel<<<1,1,ALLOC_SIZE*sizeof(int)>>>(offset);

    cudaDeviceReset();
    return 0;
}
