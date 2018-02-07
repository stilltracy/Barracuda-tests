#include <limits.h>
#include <stdio.h>

__global__ void oversized_alloc_kernel()
{
  char* devMem = (char*)malloc(ULONG_MAX);
  free(devMem);
}

int main()
{
  cudaThreadSetLimit(cudaLimitMallocHeapSize, 128*sizeof(char));
  oversized_alloc_kernel<<<1, 1>>>();

  cudaDeviceReset();
  return 0;
}
