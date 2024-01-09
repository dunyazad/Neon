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
			Eigen::Vector3f p[8];
			float val[8];
		} GRIDCELL;

		typedef struct {
			Eigen::Vector3f p[3];
		} TRIANGLE;
	}

	void Test();

	class TSDF
	{
	public:
		TSDF();
		TSDF(float voxelSize, const Eigen::Vector3f& minPoint, const Eigen::Vector3f& maxPoint);

		//void Apply(Neon::Mesh* mesh);

		void IntegrateWrap(const std::vector<glm::vec3>& vertices, const Eigen::Matrix4f& transform, const Eigen::Vector3f& vmin, const Eigen::Vector3f& vmax, int rows, int columns);
		void Integrate(const thrust::device_vector<Eigen::Vector3f>& vertices, const Eigen::Matrix4f& transform, const Eigen::Vector3f& vmin, const Eigen::Vector3f& vmax, int rows, int columns);

		void UpdateValues();

		void BuildGridCells(float isoValue);

		void TestValues(Neon::Scene* scene);
		void TestTriangles(Neon::Scene* scene);

	protected:
		float voxelSize;
		Eigen::Vector3f minPoint;
		Eigen::Vector3f maxPoint;
		Eigen::Vector3f centerPoint;

		size_t voxelCountX;
		size_t voxelCountY;
		size_t voxelCountZ;

		thrust::device_vector<float> values;
		thrust::device_vector<Eigen::Vector3f> positions;
		thrust::device_vector<MarchingCubes::GRIDCELL> gridcells;
		thrust::device_vector<MarchingCubes::TRIANGLE> triangles;
	};
}
