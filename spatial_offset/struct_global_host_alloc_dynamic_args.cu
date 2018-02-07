#include <stdio.h>
#include <unistd.h>

struct IntSandwich {
  int beginning;
  int middle[1];
  int end;  
};

__global__ void access_offset_kernel(struct IntSandwich *hostMem, int offset) {
#ifdef R
    volatile int i = hostMem->middle[offset];
#elif W
    hostMem->middle[offset] = 42;
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

    struct IntSandwich *hostMem;
    cudaMalloc((void**)&hostMem, sizeof(struct IntSandwich));
    access_offset_kernel<<<1,1>>>(hostMem, offset);
    cudaFree(hostMem);

    cudaDeviceReset();
    return 0;
}
