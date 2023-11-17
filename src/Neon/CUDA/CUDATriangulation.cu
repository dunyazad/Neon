#include "CUDACommon.cuh"
#include <Neon/CUDA/CUDATriangulation.h>

namespace NeonCUDA
{
#pragma region commentout
	//__device__ void lock(int* mutex) {
	//	printf("Lock\n");

	//	//while (atomicCAS(mutex, 0, 1) != 0);
	//	while (atomicExch(mutex, 1) != 0)
	//	{
	//		printf("waiting\n");
	//	}
	//	
	//	printf("After Lock\n");
	//}

	//__device__ void unlock(int* mutex) {
	//	atomicExch(mutex, 0);
	//}

	//namespace MemoryPool
	//{
	//	__device__ __managed__ char* allocatedMemory = nullptr;
	//	__device__ __managed__ char* currentMemory = nullptr;
	//	__device__ __managed__ size_t allocatedSize = 0;
	//	__device__ __managed__ int mutex = 0;

	//	__host__ bool Initialize(size_t size)
	//	{
	//		auto result = cudaMalloc((void**)&allocatedMemory, size);
	//		CUDA_CHECK_ERROR(result);

	//		currentMemory = allocatedMemory;
	//		allocatedSize = size;

	//		printf("MemoryPool is successfully initialized.\n");

	//		return true;
	//	}

	//	__host__ bool Terminate()
	//	{
	//		auto result = cudaFree(allocatedMemory);
	//		CUDA_CHECK_ERROR(result);

	//		result = cudaFree(&mutex);
	//		CUDA_CHECK_ERROR(result);

	//		printf("MemoryPool is successfully terminated.\n");

	//		return true;
	//	}

	//	template<typename T>
	//	__device__ T* New()
	//	{
	//		printf("New()\n");

	//		lock(&mutex);

	//		printf("New()  111111111111\n");

	//		auto output = currentMemory;
	//		currentMemory += sizeof(T);

	//		unlock(&mutex);

	//		return (T*)output;
	//	}
	//}

	//__global__ void Test()
	//{
	//	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	//	if (threadId < MemoryPool::allocatedSize / sizeof(int))
	//	{
	//		printf("[%d] ", threadId);

	//		auto value = MemoryPool::New<int>();

	//		printf("11111111\n");

	//		printf("value = %d\n", (int)value - (int)MemoryPool::allocatedMemory);
	//	}

	//	//printf("AllocatedSize: %d\n", MemoryPool::allocatedSize / sizeof(int));

	//	//for (size_t i = 0; i < MemoryPool::allocatedSize / sizeof(int); i++)
	//	//{
	//	//	auto value = MemoryPool::New<int>();

	//	//	//printf("[%d] value = %d\n", i, (int)value - (int)MemoryPool::allocatedMemory);
	//	//}
	//}

	//void MemoryPoolTest()
	//{
	//	MemoryPool::Initialize(102400000);

	//	Test<<<1,10>>>();

	//	cudaDeviceSynchronize();

	//	CUDA_CHECK_ERROR(cudaGetLastError());

	//	MemoryPool::Terminate();
	//}
#pragma endregion

__device__ bool __device__IsPointInTriangle(const glm::vec3& point, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C) {
	// Compute barycentric coordinates
	glm::vec3 v0 = B - A;
	glm::vec3 v1 = C - A;
	glm::vec3 v2 = point - A;

	float dot00 = glm::dot(v0, v0);
	float dot01 = glm::dot(v0, v1);
	float dot02 = glm::dot(v0, v2);
	float dot11 = glm::dot(v1, v1);
	float dot12 = glm::dot(v1, v2);

	// Compute barycentric coordinates
	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is inside the triangle
	return (u >= 0) && (v >= 0) && (u + v <= 1);
}

//__global__ void __global__SplitTriangles(DMesh* mesh)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	if (tid < mesh->n_triangles)
//	{
//		auto t = mesh->triangles[tid];
//		if (0 == t.x)
//			return;
//
//		printf("[tid: %d] t : %d, %d, %d, %d\n", tid, t.x, t.y, t.z, t.w);
//		auto v0 = mesh->points[t.y];
//		auto v1 = mesh->points[t.z];
//		auto v2 = mesh->points[t.w];
//
//		//printf("v0: %.2f, %.2f, %.2f, v0: %.2f, %.2f, %.2f, v0: %.2f, %.2f, %.2f\n", v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
//
//		for (size_t i = 0; i < mesh->n_points; i++)
//		{
//			if (__device__IsPointInTriangle(mesh->points[i], v0, v1, v2))
//			{
//				mesh->new_triangles[tid * 3 + 0] = glm::ivec4(1, t.y, t.z, i);
//				mesh->new_triangles[tid * 3 + 1] = glm::ivec4(1, t.z, t.w, i);
//				mesh->new_triangles[tid * 3 + 2] = glm::ivec4(1, t.w, t.y, i);
//
//				atomicAdd(&mesh->n_new_triangles, 3);
//
//				mesh->deleted_triangles[tid] = i;
//
//				atomicAdd(&mesh->n_deleted_triangles, 1);
//			}
//		}
//	}
//}

__device__ void __device__SplitTriangles(DMesh* mesh)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < mesh->n_triangles)
	{
		auto t = mesh->triangles[tid];
		if (0 == t.x)
			return;

		//printf("[tid: %d] t : %d, %d, %d, %d\n", tid, t.x, t.y, t.z, t.w);
		auto v0 = mesh->points[t.y];
		auto v1 = mesh->points[t.z];
		auto v2 = mesh->points[t.w];

		//printf("v0: %.2f, %.2f, %.2f, v0: %.2f, %.2f, %.2f, v0: %.2f, %.2f, %.2f\n", v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);

		for (size_t i = 0; i < mesh->n_points; i++)
		{
			if (__device__IsPointInTriangle(mesh->points[i], v0, v1, v2))
			{
				atomicAdd(&mesh->n_new_triangles, 3);

				mesh->new_triangles[(mesh->n_new_triangles - 3) + 0] = glm::ivec4(1, t.y, t.z, i);
				mesh->new_triangles[(mesh->n_new_triangles - 3) + 1] = glm::ivec4(1, t.z, t.w, i);
				mesh->new_triangles[(mesh->n_new_triangles - 3) + 2] = glm::ivec4(1, t.w, t.y, i);

				atomicAdd(&mesh->n_deleted_triangles, 1);
				mesh->deleted_triangles[mesh->n_deleted_triangles - 1] = i;
				mesh->triangles[mesh->n_deleted_triangles - 1].x = 0;

				return;
			}
		}
	}
}

__global__ void __global__RefreshMesh(DMesh* mesh)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	printf("[tid : %d] gridDim.x : %d, blockDim.x : %d\n", tid, gridDim.x, blockDim.x);

	for (size_t i = 0; i < mesh->n_new_triangles; i++)
	{
		mesh->triangles[mesh->n_triangles + i] = mesh->new_triangles[i];
		mesh->new_triangles[i] = glm::ivec4(0, 0, 0, 0);
	}

	mesh->n_triangles += mesh->n_new_triangles;

	mesh->n_new_triangles = 0;
}

__global__ void __global__PrintMesh(DMesh* mesh)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	printf("\nMesh\n");
	
	printf("Number of Points : %d\n", mesh->n_points);
	/*for (size_t i = 0; i < mesh->n_points; i++)
	{
		printf("%.2f, %.2f, %.2f\n", mesh->points[i].x, mesh->points[i].y, mesh->points[i].z);
	}*/

	printf("Number of Triangles : %d\n", mesh->n_triangles);
	for (size_t i = 0; i < mesh->n_triangles; i++)
	{
		printf("%d, %d, %d, %d\n", mesh->triangles[i].x, mesh->triangles[i].y, mesh->triangles[i].z, mesh->triangles[i].w);
	}
	
	printf("Number of New Triangles : %d\n", mesh->n_new_triangles);
	for (size_t i = 0; i < mesh->n_new_triangles; i++)
	{
		printf("%d %d %d %d\n", mesh->new_triangles[i].x, mesh->new_triangles[i].y, mesh->new_triangles[i].z, mesh->new_triangles[i].w);
	}

	printf("Number of Deleted Triangles : %d\n", mesh->n_deleted_triangles);
	for (size_t i = 0; i < mesh->n_deleted_triangles; i++)
	{
		printf("%d\n", mesh->deleted_triangles[i]);
	}
}

__global__ void __global__Triangulate(DMesh* mesh)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__device__SplitTriangles(mesh);

	//printf("[tid : %d] gridDim.x : %d, blockDim.x : %d\n", tid, gridDim.x, blockDim.x);

	if (1 == tid)
	{
		__global__RefreshMesh<<<1, 1>>>(mesh);

		__global__PrintMesh<<<1, 1>>>(mesh);
	}
}


void GetSupraTriangle(const std::vector<glm::vec3>& points, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2)
{
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float X = 0.0f;
	float Y = 0.0f;
	float Z = 0.0f;

	for (auto& p : points)
	{
		x = min(x, p.x);
		X = max(X, p.x);
		y = min(y, p.y);
		Y = max(Y, p.y);
		z = min(z, p.z);
		Z = max(Z, p.z);
	}

	float cx = (x + X) * 0.5f;
	float cy = (y + Y) * 0.5f;
	float cz = (z + Z) * 0.5f;

	float sx = (x - cx) * 3 + x;
	float sy = (y - cy) * 3 + y;
	float sz = (z - cz) * 3 + z;
	float sX = (X - cx) * 3 + X;
	float sY = (Y - cy) * 3 + Y;
	float sZ = (Z - cz) * 3 + Z;

	p0 = glm::vec3(sx, sy, 0.0f);
	p1 = glm::vec3(sX, sy, 0.0f);
	p2 = glm::vec3(cx, sY, 0.0f);
}

std::vector<glm::ivec3> Triangulate(const std::vector<glm::vec3>& points)
{
	vector<glm::vec3> pts(points);

	vector<glm::ivec3> result;

	{ // Get Supra triangle as a initial triangle
		glm::vec3 sp0, sp1, sp2;
		GetSupraTriangle(points, sp0, sp1, sp2);

		pts.push_back(sp0);
		pts.push_back(sp1);
		pts.push_back(sp2);
	}

	DMesh mesh;
	DMesh* device_mesh = nullptr;
	auto supratriangle = glm::ivec4(1, pts.size() - 3, pts.size() - 2, pts.size() - 1);

	{ // Alloc and copy data from host to device
		auto e = cudaMalloc(&mesh.points, sizeof(glm::vec3) * pts.size());
		CUDA_CHECK_ERROR(e);
		e = cudaMemcpy(mesh.points, &pts[0], sizeof(glm::vec3) * pts.size(), cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR(e);

		e = cudaMalloc(&mesh.triangles, sizeof(glm::ivec4) * (pts.size() + 2));
		CUDA_CHECK_ERROR(e);
		e = cudaMemset(mesh.triangles, 0, sizeof(glm::ivec4) * (pts.size() + 2));
		CUDA_CHECK_ERROR(e);

		e = cudaMemcpy(mesh.triangles, &supratriangle[0], sizeof(glm::ivec4), cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR(e);

		e = cudaMalloc(&mesh.new_triangles, (sizeof(glm::ivec4) * (pts.size() + 2) * 3));
		CUDA_CHECK_ERROR(e);
		e = cudaMemset(mesh.new_triangles, 0, (sizeof(glm::ivec4) * (pts.size() + 2) * 3));
		CUDA_CHECK_ERROR(e);

		e = cudaMalloc(&mesh.deleted_triangles, sizeof(size_t) * (pts.size() + 2) * 3);
		CUDA_CHECK_ERROR(e);
		e = cudaMemset(mesh.deleted_triangles, 0, sizeof(size_t) * (pts.size() + 2) * 3);
		CUDA_CHECK_ERROR(e);

		mesh.n_points = points.size();
		mesh.n_triangles = 1;
		mesh.n_new_triangles = 0;
		mesh.n_deleted_triangles = 0;

		e = cudaMalloc(&device_mesh, sizeof(DMesh));
		CUDA_CHECK_ERROR(e);
		e = cudaMemcpy(device_mesh, &mesh, sizeof(DMesh), cudaMemcpyHostToDevice);
		CUDA_CHECK_ERROR(e);
	}

	auto threadCount = (mesh.n_points + 2) * 3;
	auto blockCount = threadCount / 1024 + 1;

	auto t1 = Neon::Time("Triangulate1");
	__global__Triangulate<<<blockCount, 1024>>>(device_mesh);
	t1.Stop();

	auto t2 = Neon::Time("Triangulate2");
	__global__Triangulate<<<blockCount, 1024>>>(device_mesh);
	t2.Stop();

	//auto t3 = Neon::Time("Triangulate3");
	//__global__Triangulate << <blockCount, 1024 >> > (device_mesh);
	//t3.Stop();

	{
		glm::ivec4* triangles = new glm::ivec4[sizeof(glm::ivec4) * (pts.size() + 2)];
		auto e = cudaMemcpy(triangles, mesh.triangles, sizeof(glm::ivec4) * (pts.size() + 2), cudaMemcpyDeviceToHost);
		CUDA_CHECK_ERROR(e);

		DMesh mesh;
		e = cudaMemcpy(&mesh, device_mesh, sizeof(DMesh), cudaMemcpyDeviceToHost);
		CUDA_CHECK_ERROR(e);

		printf("not : %d\n", mesh.n_triangles);
		for (size_t i = 0; i < mesh.n_triangles; i++)
		{
			printf("%d, %d, %d, %d\n", triangles[i].x, triangles[i].y, triangles[i].z, triangles[i].w);
			if (1 == triangles[i].x)
			{
				result.push_back(glm::ivec3(triangles[i].y, triangles[i].z, triangles[i].w));
			}
		}
	}

	return result;
}
}
