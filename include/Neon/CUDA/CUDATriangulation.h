#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/CUDA/CUDACommon.h>
using namespace std;

namespace NeonCUDA
{
	struct DMesh
	{
		size_t n_points = 0;
		glm::vec3* points = nullptr;

		size_t n_triangles = 0;
		glm::ivec4* triangles = nullptr;
		int triangles_lock = 0;
		
		size_t n_new_triangles = 0;
		glm::ivec4* new_triangles = nullptr;
		int new_triangles_lock = 0;
		
		size_t n_deleted_triangles = 0;
		size_t* deleted_triangles = nullptr;
		int deleted_triangles_lock = 0;

		int new_triangle_created = 0;
	};

	__global__ void __global__SplitTriangles(DMesh* mesh);

	__global__ void __global__Triangulate(DMesh* mesh);

	void GetSupraTriangle(const std::vector<glm::vec3>& points, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2);
	vector<glm::ivec3> Triangulate(const vector<glm::vec3>& points);
}
