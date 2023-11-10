#include "CUDACommon.cuh"
#include <Neon/CUDA/CUDAList.h>

namespace NeonCUDA {

	struct ListElement
	{
		ListElement* previous = nullptr;
		ListElement* next = nullptr;
		size_t id = 0;
		size_t data;
		bool isInUse = false;
	};

	//__device__ size_t count = 0;

	class ListElementPool
	{
	public:
		ListElement* memory = nullptr;
		ListElement* availableBlock = nullptr;
		size_t totalAllocated = 0;
		size_t count = 0;

		bool Initialize(size_t size)
		{
			{
				cudaError_t cudaStatus = cudaMalloc((void**)&memory, size * sizeof(ListElement));

				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
					return false;
				}
			}

			{
				cudaError_t cudaStatus = cudaMemset((void*)memory, 0, size * sizeof(ListElement));

				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
					return false;
				}
			}

			availableBlock = memory;

			return true;
		}

		__device__ ListElement* New()
		{
			printf("ListElement* New()\n");

			auto result = availableBlock;
			count = atomicAdd(&count, 1);
			printf("11111111111\n");
			result->id = count;
			printf("22222222222\n");
			availableBlock += sizeof(ListElement);
			printf("33333333333\n");

			printf("count : %d\n", count);

			return result;
		}
	};

	__global__ void Fill(ListElement* memory, size_t size)
	{
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId < size)
		{
			if (threadId > 0)
			{
				memory[threadId].previous = &memory[threadId - 1];
			}
			memory[threadId].data = threadId;

			if (threadId < size - 1)
			{
				memory[threadId].next = &memory[threadId + 1];
			}
		}
	}

	__global__ void FillFromDevice(ListElementPool* dev_pool, size_t size)
	{
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId < size)
		{
			printf("threadId = %d\n", threadId);
			
			auto element = dev_pool->New();
			//auto element = dev_pool.New();

			printf("element->id = %d\n", element->id);

			element->data = threadId+1;

			printf("element->data = %d\n", element->data);
		}
	}

	__global__ void Temp(ListElement* p, int* result)
	{
		for (size_t i = 0; i < 20; i++)
		{
			result[i] = p->data;
			p = p->next;
		}
	}

	void ListTestFunction()
	{
		const size_t size = 12000000;

		ListElementPool pool;
		pool.Initialize(size * 10);

		ListElementPool* dev_pool;
		cudaMalloc(&dev_pool, sizeof(ListElementPool));
		cudaMemcpy(dev_pool, &pool, sizeof(ListElementPool), cudaMemcpyHostToDevice);

		//Fill<<<48000, 250>>>(pool.memory, size);
		FillFromDevice<<<1, 10>>>(dev_pool, size);

		//cudaMemcpy(&pool, dev_pool, size * sizeof(ListElementPool), cudaMemcpyDeviceToHost);

		cudaMemcpy(&pool, dev_pool, sizeof(ListElementPool), cudaMemcpyDeviceToHost);

		ListElement* host = new ListElement[size];
		cudaMemcpy(host, pool.memory, size * sizeof(ListElement), cudaMemcpyDeviceToHost);
		//for (size_t i = 0; i < 100; i++)
		//{
		//	printf("[%d].data : %d\n", i, host[i].id);
		//}

		//int* dev_result;
		//cudaMalloc(&dev_result, sizeof(int) * 20);

		//Temp<<<1,1>>>(pool.memory, dev_result);

		//int* host_result = new int[20];
		//cudaMemcpy(host_result, dev_result, sizeof(int) * 20, cudaMemcpyDeviceToHost);

		//for (size_t i = 0; i < 20; i++)
		//{
		//	std::cout << "(" << i << ") " << host_result[i] << std::endl;
		//}



		//for (size_t i = 0; i < size; i++)
		//{
		//	std::cout << "host[" << i << "] data : " << host[i].previous << std::endl;
		//	if (20 == i)
		//	{
		//		break;
		//	}
		//}
	}

	template<typename T>
	class ObjectPool
	{
	public:
		ObjectPool()
		{
		}

		~ObjectPool()
		{
			Terminate();
		}

		void Initialize(size_t count)
		{
			if (false == initialized)
			{
				auto result = cudaMalloc((void**)&objects, sizeof(T) * count);
				CUDA_CHECK_ERROR(result);

				current = objects;

				result = cudaMalloc((void**)&device_pool, sizeof(ObjectPool<T>));
				CUDA_CHECK_ERROR(result);

				result = cudaMemcpy(device_pool, this, sizeof(ObjectPool<T>), cudaMemcpyDeviceToHost);

				printf("ObjectPool successfully initialized.\n");
			}
		}

		void Terminate()
		{
			if (false == terminated)
			{
				auto result = cudaFree(objects);
				CUDA_CHECK_ERROR(result);

				printf("ObjectPool successfully terminated.\n");
			}
		}

		__device__ T* New()
		{
			if (0 == allocated)
			{
				current = objects;
			}

			current += sizeof(T);
			allocated++;
			return current - sizeof(T);
		}

		__host__ __device__ inline T* Objects() { return objects; }
		__host__ __device__ inline ObjectPool<T>* DevicePool() { return device_pool; }

	//private:
		ObjectPool<T>* device_pool = nullptr;

		T* objects = nullptr;
		T* current = nullptr;
		bool initialized = false;
		bool terminated = false;
		size_t allocated = 0;
	};

	/*__global__ void Alloc(ObjectPool<ListElement>* device_pool)
	{
		for (size_t i = 0; i < 100; i++)
		{
			printf("* %d\n", i);

			auto element = device_pool->New();

			printf("! %d\n", i);

			element->data = i + 1;

			CUDA_CHECK_ERROR(cudaGetLastError());

			printf("- %d\n", i);

			printf("element->data : %d\n", element->data);
		}
	}

	void ObjectPoolTest()
	{
		ObjectPool<ListElement> pool;
		pool.Initialize(1024000);

		Alloc<<<1, 1>>>(pool.DevicePool());
		CUDA_CHECK_ERROR(cudaGetLastError());

		ListElement* host = new ListElement[110];
		cudaMemcpy(host, pool.Objects(), sizeof(ListElement) * 100, cudaMemcpyDeviceToHost);

		for (size_t i = 0; i < 110; i++)
		{
			std::cout << "ListElement[" << i << "] : " << host[i].data << std::endl;
		}
	}*/

}
