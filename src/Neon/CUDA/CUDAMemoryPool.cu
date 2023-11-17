#include "CUDACommon.cuh"
#include <Neon/CUDA/CUDAMemoryPool.h>

namespace NeonCUDA
{
	typedef size_t id_t;
	struct Node
	{
		size_t is_using = 0;
		id_t previous = -1;
		id_t next = -1;
		size_t data = -1;
	};

	struct MemoryPool
	{
		Node* allocated = nullptr;
		size_t allocatedCount = 0;
		id_t base = 0;
		id_t current = 0;
	};

	struct vertex
	{
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;
		id_t triangle_indices = -1;
	};

	__device__ void __device__lock(int* lock) {
		while (atomicExch(lock, 1) != 0);
	}

	__device__ void __device__unlock(int* lock) {
		atomicExch(lock, 0);
	}

	__device__ Node* __device__GetNode(MemoryPool* memory_pool, id_t p)
	{
		return &memory_pool->allocated[p];
	}

	__device__ id_t New(MemoryPool* memory_pool)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		id_t p = atomicAdd(&memory_pool->current, 1);
		memory_pool->allocated[p].is_using = 1;

		return p;
	}

	__global__ void AllocNodes(MemoryPool* memory_pool)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		auto ptr = New(memory_pool);
		auto node = __device__GetNode(memory_pool, ptr);
		node->data = tid;
	}

	void MemoryPoolTest()
	{
		size_t count = 120000000;

		MemoryPool memory_pool;

		MemoryPool* device_memory_pool = nullptr;

		auto e = cudaMalloc(&memory_pool.allocated, sizeof(Node) * count);
		CUDA_CHECK_ERROR(e);

		memory_pool.allocatedCount = count;

		e = cudaMalloc(&device_memory_pool, sizeof(MemoryPool));
		CUDA_CHECK_ERROR(e);

		e = cudaMemcpy(device_memory_pool, &memory_pool, sizeof(MemoryPool), cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR(e);

		auto threadCount = count;
		auto blockCount = threadCount / 1024 + 1;
		{
			auto t = Neon::Time("AllocNodes");
			AllocNodes<<<blockCount, 1024>>>(device_memory_pool);
			auto e = cudaGetLastError();
			CUDA_CHECK_ERROR(e);
		}

		e = cudaMemcpy(&memory_pool, device_memory_pool, sizeof(MemoryPool), cudaMemcpyDeviceToHost);
		CUDA_CHECK_ERROR(e);

		Node* host_allocated = new Node[count];
		e = cudaMemcpy(host_allocated, memory_pool.allocated, sizeof(Node) * count, cudaMemcpyDeviceToHost);
		CUDA_CHECK_ERROR(e);

		for (size_t i = 0; i < count; i++)
		{
			auto n = host_allocated[i];
			if (n.is_using)
			{
				printf("[%d] is_using: %d, data: %d\n", i, n.is_using, n.data);
			}
		}
	}
}
