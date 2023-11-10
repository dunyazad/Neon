#pragma once

#include <Neon/CUDA/CUDACommon.h>

namespace NeonCUDA
{
	void ListTestFunction();

	void ObjectPoolTest();

	//__device__ bool AllocDeviceMemory(char** memory, size_t size);

	//template<typename T>
	//class ListElement
	//{
	//public:
	//	__device__ ListElement(const T& t) : t(t) {}
	//	__device__ ~ListElement() {}

	//	ListElement<T>* previous = nullptr;
	//	ListElement<T>* next = nullptr;

	//	__device__ static ListElement<T>* New()
	//	{
	//		if (false == initialized)
	//		{
	//			initialized = Initialize();
	//		}

	//		ListElement<T>* allocated = (ListElement<T>*)availableBlock;
	//		allocated->inUse = true;
	//		return allocated;
	//	}

	//	__device__ static bool Initialize(size_t size = sizeof(ListElement<T>) * 120000)
	//	{
	//		auto result = AllocDeviceMemory(&memory, size);
	//		availableBlock = memory;
	//		return result;
	//	}

	//private:
	//	bool inUse = false;
	//	T t;

	//	inline static char* memory = nullptr;
	//	inline static char* availableBlock = nullptr;
	//	inline static size_t totalAllocated = 0;
	//	inline static bool initialized = false;
	//};

	//template<typename T>
	//class List
	//{
	//public:
	//	__device__ List()
	//	{
	//	}

	//	__device__ ~List()
	//	{
	//	}

	//	__device__ void push_back(const T& t)
	//	{
	//		if (0 == elementCount)
	//		{
	//			head = ListElement<T>::New();
	//			tail = head;
	//		}
	//		else
	//		{

	//		}

	//		elementCount++;
	//	}

	//private:
	//	ListElement<T>* head = nullptr;
	//	ListElement<T>* tail = nullptr;
	//	size_t elementCount = 0;
	//};
}
