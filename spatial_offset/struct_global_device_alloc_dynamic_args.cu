#include <stdio.h>
#include <unistd.h>

struct IntSandwich {
  int beginning;
  int middle[1];
  int end;  
};

__global__ void access_offset_kernel(int offset) {
    struct IntSandwich *devMem = (struct IntSandwich*)malloc(sizeof(struct IntSandwich));
#ifdef R
    volatile int i = devMem->middle[offset];
#elif W
    devMem->middle[offset] = 42;
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

    cudaThreadSetLimit(cudaLimitMallocHeapSize, 2*sizeof(struct IntSandwich));
    access_offset_kernel<<<1,1>>>(offset);

    cudaDeviceReset();
    return 0;
}
