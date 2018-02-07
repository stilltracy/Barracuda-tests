#include <stdio.h>
#include <unistd.h>

#define ALLOC_SIZE 1024

__global__ void access_offset_kernel(int offset) {
    int devMem[ALLOC_SIZE];
    devMem[0] = 0; devMem[1] = devMem[0]; // for init/unused warnings
    
    if (offset >= 0)
        offset += (ALLOC_SIZE-1);
    // this slight difference in the conditional produced different results 
    // so we include one version for posterity to investigate
#ifdef R
    volatile int i = devMem[offset];
#elif W
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
  
    access_offset_kernel<<<1,1>>>(offset);

    cudaDeviceReset();
    return 0;
}
