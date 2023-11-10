#include "CUDACommon.cuh"
#include <Neon/CUDA/CUDAMemoryPool.h>

namespace NeonCUDA
{
	// Hypothetical CUDA Mutex type
	typedef struct {
		int data;  // Data used for mutex implementation
	} cudaMutex_t;

	// Hypothetical CUDA Mutex functions
	__device__ void cudaLockMutex(cudaMutex_t* mutex) {
		printf("cudaLockMutex\n");

		printf("mutex->data : %d\n", mutex->data);

		// Implementation of locking mechanism
		// This is a simplified example and may not be suitable for production use
		while (atomicExch(&(mutex->data), 1) != 0) {
			// Spin until the lock is 
			//printf("In loop : %d\n", mutex->data);
		}

		printf("Locked : %d\n", mutex->data);
	}

	__device__ void cudaUnlockMutex(cudaMutex_t* mutex) {
		// Implementation of unlocking mechanism
		// This is a simplified example and may not be suitable for production use
		atomicExch(&(mutex->data), 0);
	}

	namespace MemoryPool
	{
		__device__ __managed__ char* allocatedMemory = nullptr;
		__device__ __managed__ char* currentMemory = nullptr;
		__device__ __managed__ size_t allocatedSize = 0;
		__device__ __managed__ cudaMutex_t* mutex = nullptr;

		__host__ bool Initialize(size_t size)
		{
			auto result = cudaMalloc((void**)&allocatedMemory, size);
			CUDA_CHECK_ERROR(result);

			currentMemory = allocatedMemory;
			allocatedSize = size;

			result = cudaMalloc((void**)&mutex, sizeof(cudaMutex_t));
			CUDA_CHECK_ERROR(result);

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
			printf("New()\n");

			cudaLockMutex(mutex);

			printf("New() 11111111\n");

			auto output = currentMemory;
			currentMemory += sizeof(T);
			cudaUnlockMutex(mutex);
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

			printf("11111111\n");

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

		Test<<<1,10>>>();

		CUDA_CHECK_ERROR(cudaGetLastError());

		MemoryPool::Terminate();
	}
}
