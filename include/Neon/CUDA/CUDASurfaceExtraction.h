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
	class SurfaceExtractor
	{
	public:
		Neon::Scene* scene;

	public:
		SurfaceExtractor(size_t hResolution = 256, size_t vResolution = 480, float voxelSize = 0.1f);
		~SurfaceExtractor();

		void PrepareNewFrame();

		void BuildDepthMap(const thrust::device_vector<Eigen::Vector3f>& inputPoints);

		void NewFrameWrapper(Neon::Scene* scene, Neon::Mesh* mesh, const Eigen::Matrix4f& transformMatrix);
		void NewFrame(const thrust::device_vector<Eigen::Vector3f>& inputPoints, const Eigen::AlignedBox3f& aabb, const Eigen::Matrix4f& transformMatrix);

		void Initialize();

	protected:

		// y = 220

		size_t hResolution = 256;
		size_t vResolution = 480;
		float xUnit = 0.1f;
		float yUnit = 0.1f;
		float voxelSize = 0.1f;

		thrust::device_vector<float> depthMap;
		thrust::device_vector<float> voxelValues;
		thrust::device_vector<Eigen::Vector3f> voxelCenterPositions;
		thrust::device_vector<GLuint> meshIndices;
		thrust::device_vector<MarchingCubes::TRIANGLE> triangles;

		Eigen::AlignedBox3f lastFrameAABB;
		size_t voxelCountX = 0;
		size_t voxelCountY = 0;
		size_t voxelCountZ = 0;
	};

	void BuildDepthMapTest(Neon::Scene* scene, Neon::Mesh* mesh, Eigen::Matrix4f& transform);
}
