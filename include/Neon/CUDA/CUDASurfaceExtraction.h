#pragma once

typedef unsigned int GLuint;

#include <Neon/CUDA/CUDACommon.h>

namespace Neon
{
	class Mesh;
	class Scene;
}

namespace NeonCUDA
{
	void DoSurfaceExtractionWrapper(Neon::Scene* scene, Neon::Mesh* mesh, const Eigen::Matrix4f& transform);
	void DoSurfaceExtraction(Neon::Scene* scene, const thrust::device_vector<Eigen::Vector3f>& inputPoints, const Eigen::AlignedBox3f& aabb, const Eigen::Matrix4f& transform);

	class SurfaceExtractor
	{
	public:
		Neon::Scene* scene;

	public:
		SurfaceExtractor(size_t hResolution = 256, size_t vResolution = 480, float voxelSize = 0.1f);
		~SurfaceExtractor();

		void PrepareNewFrame();

		void NewFrameWrapper(Neon::Scene* scene, Neon::Mesh* mesh, const Eigen::Matrix4f& transform);
		void NewFrame(const thrust::device_vector<Eigen::Vector3f>& inputPoints, const Eigen::AlignedBox3f& aabb, const Eigen::Matrix4f& transform);

		void Initialize();

	protected:
		size_t hResolution = 256;
		size_t vResolution = 480;
		float xUnit = 0.1f;
		float yUnit = 0.1f;
		float voxelSize = 0.1f;

		thrust::device_vector<float> voxelValues;
		thrust::device_vector<Eigen::Vector3f> voxelCenterPositions;
		thrust::device_vector<GLuint> meshIndices;

		Eigen::AlignedBox3f lastFrameAABB;
		size_t voxelCountX = 0;
		size_t voxelCountY = 0;
		size_t voxelCountZ = 0;
	};
}
