#include <stdio.h>

__global__ void hello(void) {
	printf("hello CUDA\n");
	printf("hello CUDA %d\n", threadIdx.x);
	printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

void Hello()
{
	hello<<<8,8>>>();
	cudaDeviceSynchronize(); // Important
}