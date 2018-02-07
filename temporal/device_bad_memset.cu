#include <stdio.h>

#define ALLOC_SIZE 1024

__global__ void bad_memset_kernel()
{
  int* devMem = (int*)malloc(ALLOC_SIZE*sizeof(int));
  memset(devMem, 0, (ALLOC_SIZE+1)*sizeof(int));
  free(devMem);
}

int main()
{
  cudaThreadSetLimit(cudaLimitMallocHeapSize, 2*ALLOC_SIZE*sizeof(int));
  bad_memset_kernel<<<1, 1>>>();

  cudaDeviceReset();
  return 0;
}
