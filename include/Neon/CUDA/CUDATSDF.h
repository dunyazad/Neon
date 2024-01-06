#pragma once

#include <Neon/CUDA/CUDACommon.h>

namespace Neon
{
	class Mesh;
	class Scene;
}

namespace NeonCUDA
{
	void Test();

	class TSDF
	{
	public:
		TSDF();
		TSDF(float voxelSize, const float3& minPoint, const float3& maxPoint);

		void Apply(Neon::Mesh* mesh);

		void UpdateValues();

		void Test(Neon::Scene* scene);

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
	};
}
