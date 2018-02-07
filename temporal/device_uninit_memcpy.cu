#include <stdio.h>
#define SRC_SIZE 65536
#define DST_SIZE 65536
#define CPY_SIZE 65536

__global__ void bad_memcpy_kernel()
{
  int* devMemSrc = (int*)malloc(SRC_SIZE*sizeof(int));
  int* devMemDest = (int*)malloc(DST_SIZE*sizeof(int));
  memcpy(devMemDest, devMemSrc, CPY_SIZE*sizeof(int));

  free(devMemDest);
  free(devMemSrc);
}

int main()
{
  cudaThreadSetLimit(cudaLimitMallocHeapSize, 4*CPY_SIZE*sizeof(int));
  bad_memcpy_kernel<<<1, 1>>>();

  cudaDeviceReset();
  return 0;
}
