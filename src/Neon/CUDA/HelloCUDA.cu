#include <Neon/CUDA/CUDACommon.h>

__global__ void hello(void) {
	printf("hello CUDA\n");
	printf("hello CUDA %d\n", threadIdx.x);
	printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__device__ bool CheckIntersection(const glm::vec3& ta, const glm::vec3& tb, const glm::vec3& tc, const glm::vec3& ra, const glm::vec3& rb, glm::vec3& intersection)
{
	auto rayOrigin = ta;
	auto rayDirection = glm::normalize(tb - ta);
	return glm::intersectLineTriangle(rayOrigin, rayDirection, ta, tb, tc, intersection);
}

void Hello()
{
	hello<<<8,8>>>();
	cudaDeviceSynchronize(); // Important
}
