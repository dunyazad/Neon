#pragma once

#include <Neon/CUDA/CUDACommon.h>

namespace NeonCUDA
{
	class CUDAMesh
	{
	public:
		CUDAMesh();
		~CUDAMesh();

		size_t AddVertex(const glm::vec3& v);
		size_t AddIndex(unsigned int i);
		size_t AddTriangle(unsigned int i0, unsigned int i1, unsigned int i2);

		void Upload();
		void Download();

		void Subdivide();

	//private:
		thrust::host_vector<glm::vec3> host_vertices;
		thrust::host_vector<unsigned int> host_indices;

		thrust::device_vector<glm::vec3> device_vertices;
		thrust::device_vector<unsigned int> device_indices;
	};
}