#include "CUDACommon.cuh"
#include <Neon/CUDA/CUDAMemoryPool.h>

namespace NeonCUDA
{
	__device__ void mutex_lock(unsigned int* mutex) {
		unsigned int ns = 8;
		while (atomicCAS(mutex, 0, 1) == 1) {
			__nanosleep(ns);
			if (ns < 256) {
				ns *= 2;
			}
		}
	}

	__device__ void mutex_unlock(unsigned int* mutex) {
		atomicExch(mutex, 0);
	}

	namespace MemoryPool
	{
		__device__ __managed__ char* allocatedMemory = nullptr;
		__device__ __managed__ char* currentMemory = nullptr;
		__device__ __managed__ size_t allocatedSize = 0;

		__host__ bool Initialize(size_t size)
		{
			auto result = cudaMalloc((void**)&allocatedMemory, size);
			CUDA_CHECK_ERROR(result);

			currentMemory = allocatedMemory;
			allocatedSize = size;

			printf("MemoryPool is successfully initialized.\n");

			return true;
		}

		__host__ bool Terminate()
		{
			auto result = cudaFree(allocatedMemory);
			CUDA_CHECK_ERROR(result);

			printf("MemoryPool is successfully terminated.\n");

			return true;
		}

		template<typename T>
		__device__ T* New()
		{
			auto output = currentMemory;
			currentMemory += sizeof(T);
			return (T*)output;
		}
	}

	__global__ void Test()
	{
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId < MemoryPool::allocatedSize / sizeof(int))
		{
			printf("[%d] ", threadId);

			auto value = MemoryPool::New<int>();
			printf("value = %d\n", (int)value - (int)MemoryPool::allocatedMemory);
		}

		//printf("AllocatedSize: %d\n", MemoryPool::allocatedSize / sizeof(int));

		//for (size_t i = 0; i < MemoryPool::allocatedSize / sizeof(int); i++)
		//{
		//	auto value = MemoryPool::New<int>();

		//	//printf("[%d] value = %d\n", i, (int)value - (int)MemoryPool::allocatedMemory);
		//}
	}

	void MemoryPoolTest()
	{
		MemoryPool::Initialize(102400000);

		Test<<<1,1024>>>();

		CUDA_CHECK_ERROR(cudaGetLastError());

		MemoryPool::Terminate();
	}
}
