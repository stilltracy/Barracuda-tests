#include <stdio.h>
#include <unistd.h>

struct IntSandwich {
  int beginning;
  int middle[1];
  int end;  
};

__global__ void access_offset_kernel(int offset) {
    extern __shared__ int devMem[];
    struct IntSandwich* ps = (struct IntSandwich*)devMem;
    ps->beginning = 0; ps->middle[0] = 0; ps->end = 0; 
#ifdef R
    volatile int i = ps->middle[offset];
#elif W
    ps->middle[offset] = 42;
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
    
    access_offset_kernel<<<1,1,sizeof(struct IntSandwich)>>>(offset);

    cudaDeviceReset();
    return 0;
}
