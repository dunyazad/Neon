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

	std::vector<glm::vec3> FlipXInputArray(const std::vector<glm::vec3>& input, int columns, int rows);

	void BuildDepthMapWrap(Neon::Scene* scene, Neon::Mesh* mesh, size_t hResolution, size_t vResolution, float xUnit, float yUnit);
	void BuildDepthMap(Neon::Scene* scene, Neon::Mesh* mesh, size_t hResolution, size_t vResolution, float xUnit, float yUnit, thrust::device_vector<Eigen::Vector3f>& result);

	void UpscaleDepthMap(Neon::Scene* scene, Neon::Mesh* mesh, size_t hResolution, size_t vResolution, float xUnit, float yUnit, float voxelSize, thrust::device_vector<Eigen::Vector3f>& result);

	class TSDF
	{
	public:
		TSDF();
		TSDF(float voxelSize, const Eigen::Vector3f& inputMinPoint, const Eigen::Vector3f& inputMaxPoint);

		//void Apply(Neon::Mesh* mesh);

		void IntegrateWrap(const std::vector<glm::vec3>& vertices, const Eigen::Matrix4f& transform, float width, float height, int columns, int rows);
		void Integrate(const thrust::device_vector<Eigen::Vector3f>& vertices, const Eigen::Matrix4f& transform, float width, float height, int columns, int rows);

		void UpdateValues();

		void BuildGridCells(float isoValue);

		void TestValues(Neon::Scene* scene);
		void TestTriangles(Neon::Scene* scene);

		void TestInput(Neon::Scene* scene, const Eigen::Matrix4f& transform, Neon::Mesh* mesh, const glm::vec4& color);

		void ShowInversedVoxels(Neon::Scene* scene, const Eigen::Matrix4f& transform, Neon::Mesh* mesh);
		bool ShowInversedVoxelsSingle(Neon::Scene* scene, const Eigen::Matrix4f& transform, Neon::Mesh* mesh, int singleIndex);

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

	protected:
	};
}
