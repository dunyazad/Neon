#pragma once

#include <Neon/CUDA/CUDACommon.h>

namespace Neon
{
	class Mesh;
	class Scene;
}

namespace NeonCUDA
{
	namespace MarchingCubes
	{
		typedef struct {
			float3 p[8];
			float val[8];
		} GRIDCELL;

		typedef struct {
			float3 p[3];
		} TRIANGLE;
	}

	void Test();

	class TSDF
	{
	public:
		TSDF();
		TSDF(float voxelSize, const float3& minPoint, const float3& maxPoint);

		//void Apply(Neon::Mesh* mesh);

		void Integrate(const thrust::device_vector<glm::vec3>& vertices, const glm::mat4& transform, const glm::vec3& vmin, const glm::vec3& vmax, int rows, int columns);

		void UpdateValues();

		void BuildGridCells(float isoValue);

		void TestValues(Neon::Scene* scene);
		void TestTriangles(Neon::Scene* scene);

	protected:
		float voxelSize;
		float3 minPoint;
		float3 maxPoint;
		float3 centerPoint;

		size_t voxelCountX;
		size_t voxelCountY;
		size_t voxelCountZ;

		thrust::device_vector<float> values;
		thrust::device_vector<float3> positions;
		thrust::device_vector<MarchingCubes::GRIDCELL> gridcells;
		thrust::device_vector<MarchingCubes::TRIANGLE> triangles;
	};
}
