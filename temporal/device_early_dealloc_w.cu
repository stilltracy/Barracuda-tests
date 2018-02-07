#include <stdio.h>

__global__ void device_early_dealloc_kernel()
{
  char* devMem = (char*)malloc(2*sizeof(char));
  free(devMem);
  devMem[0] = '0';
}

int main()
{
  cudaThreadSetLimit(cudaLimitMallocHeapSize, 128*sizeof(char));
  device_early_dealloc_kernel<<<1, 1>>>();

  cudaDeviceReset();
  return 0;
}
