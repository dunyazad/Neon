#include <Neon/CUDA/CUDACommon.h>

__global__ void hello(void) {
	printf("hello CUDA\n");
	printf("hello CUDA %d\n", threadIdx.x);
	printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

//__device__ bool CheckIntersection(const glm::vec3& ta, const glm::vec3& tb, const glm::vec3& tc, const glm::vec3& ra, const glm::vec3& rb, glm::vec3& intersection)
//{
//	auto rayOrigin = ta;
//	auto rayDirection = glm::normalize(tb - ta);
//	return glm::intersectLineTriangle(rayOrigin, rayDirection, ta, tb, tc, intersection);
//}

//void Hello()
//{
//	hello<<<8,8>>>();
//	cudaDeviceSynchronize(); // Important
//}
//
//__global__ void checkIndex()
//{
//	printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim: (%d, %d, %d)\n",
//		threadIdx.x, threadIdx.y, threadIdx.z,
//		blockIdx.x, blockIdx.y, blockIdx.z,
//		blockDim.x, blockDim.y, blockDim.z,
//		gridDim.x, gridDim.y, gridDim.z);
//}
//
//void CheckDimension()
//{
//	int nElem = 6;
//
//	dim3 block(3); // same dim3 block(3, 1, 1);
//	dim3 grid((nElem + block.x - 1) / block.x);
//
//	printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
//	printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
//
//	checkIndex<<<grid, block>>>();
//	cudaDeviceReset();
//}
