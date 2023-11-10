#pragma once

#include <Neon/CUDA/CUDACommon.h>

namespace NeonCUDA
{
	namespace MemoryPool
	{
		__host__ bool Initialize(size_t size);
		__host__ bool Terminate();

		__device__ void* New(size_t size);

		__device__ void Alloc();
	}

	void MemoryPoolTest();
}
