#pragma once

#include <Neon/CUDA/CUDACommon.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <vector>

using namespace std;

namespace Neon {
	class Mesh;
}

namespace NeonCUDA
{
	void CUDATest(unsigned int meshVsize, void* meshV, unsigned int meshIsize, void* meshI, unsigned int patchVsize, void* patchV, unsigned int patchIsize, void* patchI);

	thrust::host_vector<Eigen::Vector3i> DelaunayTriangulation_BowyerWatson(vector<Eigen::Vector3f>& inputPoints);

	thrust::host_vector<Eigen::Vector3i> DelaunayTriangulation_DivideAndConquer(vector<Eigen::Vector3f>& inputPoints);

	thrust::host_vector<Eigen::Vector3i> DelaunayTriangulation_Custom(vector<Eigen::Vector3f>& inputPoints);

	typedef int VertexIndex;

	__device__
		const float CUDA_EPSILON = 0.000001f;

	__device__ __host__
		bool almost_equal(float a, float b);

	__device__ __host__
		bool almost_equal(const Eigen::Vector2f& a, const Eigen::Vector2f& b);

	__device__ __host__
		bool almost_equal(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

	struct Edge
	{
		Eigen::Vector3f* points;
		VertexIndex v0;
		VertexIndex v1;
		bool isBad;

		__device__ __host__
			Edge()
			: points(nullptr), v0(-1), v1(-1), isBad(true) {}

		__device__ __host__
			Edge(Eigen::Vector3f* points, VertexIndex v0, VertexIndex v1)
			: points(points), v0(v0), v1(v1), isBad(false) {}

		//__device__
		//	bool containsVertex(VertexIndex v);

		//__device__
		//	bool circumCircleContains(VertexIndex v);

		__device__
			bool operator == (const Edge& t);
	};
	
	__device__ __host__
		bool almost_equal(const Edge& e0, const Edge& e1);

	struct Triangle
	{
		Eigen::Vector3f* points;
		VertexIndex v0;
		VertexIndex v1;
		VertexIndex v2;
		bool isBad;

		__device__ __host__
			Triangle()
			: points(nullptr), v0(-1), v1(-1), v2(-1), isBad(true) {}

		__device__ __host__
			Triangle(Eigen::Vector3f* points, VertexIndex v0, VertexIndex v1, VertexIndex v2)
			: points(points), v0(v0), v1(v1), v2(v2), isBad(false) {}

		__device__
			bool containsVertex(VertexIndex v);

		__device__
			bool aabbContains(VertexIndex v);

		__device__
			bool circumCircleContains(VertexIndex v);

		__device__
			bool operator == (const Triangle& t);
	};
}
