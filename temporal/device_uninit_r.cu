#include <limits.h>
#include <stdio.h>

#define ALLOC_SIZE 1024

__global__ void simple_kernel()
{
  int* devMem = (int*)malloc(ALLOC_SIZE*sizeof(int));
  int i = devMem[0];
  i = i*i; // for unreferenced warning
  free(devMem);
}

int main()
{
  cudaThreadSetLimit(cudaLimitMallocHeapSize, 2*ALLOC_SIZE*sizeof(int));
  simple_kernel<<<1, 1>>>();

  cudaDeviceReset();
  return 0;
}
