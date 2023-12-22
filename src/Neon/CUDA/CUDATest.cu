#include <Neon/CUDA/CUDATest.h>

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonMesh.h>

#include <Neon/CUDA/CUDACommon.h>
//#include <nvtx3/nvToolsExt.h>

namespace NeonCUDA
{
	//__device__ float operator - (const float& a, const float& b) {
	//	float result;
	//	result.x = a.x - b.x;
	//	result.y = a.y - b.y;
	//	result.z = a.z - b.z;
	//	return result;
	//}

	//__device__ float cross(const float& a, const float& b) {
	//	float result;
	//	result.x = a.y * b.z - a.z * b.y;
	//	result.y = a.z * b.x - a.x * b.z;
	//	result.z = a.x * b.y - a.y * b.x;
	//	return result;
	//}

	//__device__ float dot(const float& a, const float& b) {
	//	return a.x * b.x + a.y * b.y + a.z * b.z;
	//}

	//__device__ bool PointInTriangle(const float& point, const float& v0, const float& v1, const float& v2) {
	//	float edge1 = v1 - v0;
	//	float edge2 = v2 - v0;
	//	float normal = cross(edge1, edge2);

	//	// Barycentric coordinates
	//	float detT = dot(normal, normal);
	//	float u = dot(cross(point - v2, v1 - v2), normal) / detT;
	//	float v = dot(cross(v0 - v2, point - v2), normal) / detT;
	//	float w = 1.0f - u - v;

	//	// Check if the point is inside the triangle
	//	return (u >= 0.0f) && (v >= 0.0f) && (w >= 0.0f) && (u + v + w <= 1.0f);
	//}

	__global__ void CheckPointInTriangle(unsigned int meshVsize, float* meshV, unsigned int meshIsize, GLuint* meshI, unsigned int patchVsize, float* patchV, unsigned int patchIsize, GLuint* patchI)
	{
		auto tid = threadIdx.x + blockIdx.x * blockDim.x;
		//printf("[%d] meshIsize : %d\n", tid, meshVsize);

		if (tid < meshIsize / 3)
		{
			auto i0 = meshI[tid * 3];
			auto i1 = meshI[tid * 3 + 1];
			auto i2 = meshI[tid * 3 + 2];

			auto v0x = meshV[i0];
			auto v0y = meshV[i0 + 1];
			auto v0z = meshV[i0 + 2];
			auto v1x = meshV[i1];
			auto v1y = meshV[i1 + 1];
			auto v1z = meshV[i1 + 2];
			auto v2x = meshV[i2];
			auto v2y = meshV[i2 + 1];
			auto v2z = meshV[i2 + 2];

			//printf("[%6d] %6d, %6d, %6d -> %3.6f, %3.6f, %3.6f\n", i0, i1, i2, tid, v0x, v0y, v0z);
			
			float xmin = v0x;
			float ymin = v0y;
			float zmin = v0z;
			float xmax = v0x;
			float ymax = v0y;
			float zmax = v0z;

			xmin = xmin < v1x ? xmin : v1x;
			ymin = ymin < v1y ? ymin : v1y;
			zmin = zmin < v1z ? zmin : v1z;

			xmax = xmax > v1x ? xmax : v1x;
			ymax = ymax > v1y ? ymax : v1y;
			zmax = zmax > v1z ? zmax : v1z;

			xmin = xmin < v2x ? xmin : v2x;
			ymin = ymin < v2y ? ymin : v2y;
			zmin = zmin < v2z ? zmin : v2z;

			xmax = xmax > v2x ? xmax : v2x;
			ymax = ymax > v2y ? ymax : v2y;
			zmax = zmax > v2z ? zmax : v2z;

			for (unsigned int i = 0; i < patchVsize; i++)
			{
				auto px = patchV[i * 3];
				auto py = patchV[i * 3 + 1];
				auto pz = patchV[i * 3 + 2];

				//printf("%.2f, %.2f, %.2f\n", px, py, pz);

				if (px < xmin || px > xmax) continue;
				if (py < ymin || py > ymax) continue;
				if (pz < zmin || pz > zmax) continue;

				//printf("in bounding box\n");

				//if (PointInTriangle(p, v0, v1, v2))
				//{
				//	// ºÐÇÒ
				//	printf("split\n");
				//}
			}
		}
	}

	void CUDATest(unsigned int meshVsize, void* meshVertices, unsigned int meshIsize, void* meshIndices, unsigned int patchVsize, void* patchVertices, unsigned int patchIsize, void* patchIndices)
	{
		//char* meshV = nullptr;
		//unsigned int meshVMemorySize = meshVsize * sizeof(float) * 3;
		//auto e = cudaMalloc(&meshV, meshVMemorySize);
		//CUDA_CHECK_ERROR(e);
		//e = cudaMemcpy(meshV, meshVertices, meshVMemorySize, cudaMemcpyHostToDevice);

		//char* meshI = nullptr;
		//unsigned int meshIMemorySize = meshIsize * sizeof(GLuint);
		//e = cudaMalloc(&meshI, meshIMemorySize);
		//CUDA_CHECK_ERROR(e);
		//e = cudaMemcpy(meshI, meshIndices, meshIMemorySize, cudaMemcpyHostToDevice);

		//char* patchV = nullptr;
		//unsigned int patchVMemorySize = patchVsize * sizeof(float) * 3;
		//e = cudaMalloc(&patchV, patchVMemorySize);
		//CUDA_CHECK_ERROR(e);
		//e = cudaMemcpy(patchV, patchVertices, patchVMemorySize, cudaMemcpyHostToDevice);

		//char* patchI = nullptr;
		//unsigned int patchIMemorySize = patchIsize * sizeof(GLuint);
		//e = cudaMalloc(&patchI, patchIMemorySize);
		//CUDA_CHECK_ERROR(e);
		//e = cudaMemcpy(patchI, patchIndices, patchIMemorySize, cudaMemcpyHostToDevice);

		//{
		//	auto t = Neon::Time("CheckPointInTriangle<<>>>() 1");
		//	int BlockSize = ((meshIsize / 3) / 1024) + 1;
		//	CheckPointInTriangle << <BlockSize, 1024 >> > (meshVsize, (float*)meshV, meshIsize, (GLuint*)meshI, patchVsize, (float*)patchV, patchIsize, (GLuint*)patchI);
		//}
		//{
		//	auto t = Neon::Time("CheckPointInTriangle<<>>>() 2");
		//	int BlockSize = ((meshIsize / 3) / 1024) + 1;
		//	CheckPointInTriangle << <BlockSize, 1024 >> > (meshVsize, (float*)meshV, meshIsize, (GLuint*)meshI, patchVsize, (float*)patchV, patchIsize, (GLuint*)patchI);
		//}
		//{
		//	auto t = Neon::Time("CheckPointInTriangle<<>>>() 2");
		//	int BlockSize = ((meshIsize / 3) / 1024) + 1;
		//	CheckPointInTriangle << <BlockSize, 1024 >> > (meshVsize, (float*)meshV, meshIsize, (GLuint*)meshI, patchVsize, (float*)patchV, patchIsize, (GLuint*)patchI);
		//}
		////cudaDeviceSynchronize();

		//printf("Done\n");
	}

	void Voxelize(Neon::Mesh* mesh, float voxelSize)
	{
		//auto& aabb = mesh->GetAABB();
		//float maxLength = aabb.GetXLength();
		//if (maxLength < aabb.GetYLength()) maxLength = aabb.GetYLength();
		//if (maxLength < aabb.GetZLength()) maxLength = aabb.GetZLength();

		//long voxelsPerAxis = (long)(int)ceilf(maxLength / voxelSize);

		//voxelsPerAxis = 1024;

		//nvtxRangePushA("Begin");

		//thrust::device_vector<glm::vec4> block(voxelsPerAxis * voxelsPerAxis * voxelsPerAxis);
		//thrust::fill(block.begin(), block.end(), glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));

		//nvtxRangePop();
	}

	struct Point {
		float x, y;
	};

	struct Triangle {
		int vertices[3]; // Indices of the three vertices
	};

	__host__ __device__ bool inCircle(const Point& a, const Point& b, const Point& c, const Point& d) {
		float det = (b.x - d.x) * ((c.y - d.y) * (a.x - d.x) - (c.x - d.x) * (a.y - d.y)) -
			(b.y - d.y) * ((c.x - d.x) * (a.x - d.x) - (c.y - d.y) * (a.y - d.y));
		return det > 0;
	}

	__device__ bool isCCW(const Point& a, const Point& b, const Point& c) {
		return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
	}

	//__global__ void delaunayKernel(Point* d_points, int numPoints, Triangle* d_triangles, int* d_numTriangles) {
	//	// Sort points by x-coordinate
	//	thrust::device_vector<Point> d_points_vec(d_points, d_points + numPoints);
	//	thrust::sort(d_points_vec.begin(), d_points_vec.end(),
	//		[] __device__(const Point & a, const Point & b) {
	//		return a.x < b.x;
	//	});

	//	// Initialize super triangle
	//	Triangle superTriangle;
	//	superTriangle.vertices[0] = numPoints;
	//	superTriangle.vertices[1] = numPoints + 1;
	//	superTriangle.vertices[2] = numPoints + 2;

	//	d_triangles[threadIdx.x] = superTriangle;
	//	atomicAdd(d_numTriangles, 1);

	//	__syncthreads();

	//	// Incremental Delaunay triangulation
	//	for (int i = threadIdx.x; i < numPoints; i += blockDim.x) {
	//		Point p = d_points_vec[i];

	//		for (int j = 0; j < *d_numTriangles; ++j) {
	//			Triangle currentTriangle = d_triangles[j];

	//			if (inCircle(d_points_vec[currentTriangle.vertices[0]],
	//				d_points_vec[currentTriangle.vertices[1]],
	//				d_points_vec[currentTriangle.vertices[2]], p)) {
	//				if (isCCW(d_points_vec[currentTriangle.vertices[0]],
	//					d_points_vec[currentTriangle.vertices[1]],
	//					p)) {
	//					Triangle t1, t2, t3;
	//					t1.vertices[0] = currentTriangle.vertices[0];
	//					t1.vertices[1] = currentTriangle.vertices[1];
	//					t1.vertices[2] = i;

	//					t2.vertices[0] = currentTriangle.vertices[1];
	//					t2.vertices[1] = currentTriangle.vertices[2];
	//					t2.vertices[2] = i;

	//					t3.vertices[0] = currentTriangle.vertices[2];
	//					t3.vertices[1] = currentTriangle.vertices[0];
	//					t3.vertices[2] = i;

	//					d_triangles[j] = t1;
	//					d_triangles[*d_numTriangles] = t2;
	//					d_triangles[*d_numTriangles + 1] = t3;
	//					atomicAdd(d_numTriangles, 2);
	//					break;
	//				}
	//			}
	//		}
	//	}
	//}

}
