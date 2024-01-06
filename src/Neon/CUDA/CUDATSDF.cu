#include <Neon/CUDA/CUDATSDF.h>

#include <Neon/NeonScene.h>
#include <Neon/NeonDebugEntity.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/NeonVertexBufferObject.hpp>

namespace NeonCUDA
{
	void Test()
	{

	}

#define DOT(a, b) (a).x * (b).x + (a).y * (b).y + (a).z * (b).z
#define CROSS(a, b) float3((a).y * (b).z - (b).y * (a).z, (a).z * (b).x - (b).z * (a).x, (a).x * (b).y - (b).x * (a).y)
#define LENGTHSQUARED(a) DOT((a), (a))
#define LENGTH(a) __fsqrt_rn(LENGTHSQUARED(a))
#define DISTANCESQUARED(a, b) LENGTHSQUARED((a) - (b))
#define DISTANCE(a, b) __fsqrt_rn(DISTANCESQUARED((a), (b)))
#define NORMALIZE(a) (a) / (LENGTH(a))

	__device__ __host__ float3 operator+(const float3& a, const float3& b) {
		return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	__device__ __host__ float3 operator-(const float3& a, const float3& b) {
		return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	}

	__device__ __host__ float3 operator*(float a, const float3& b) {
		return make_float3(a * b.x, a * b.y, a * b.z);
	}
	__device__ __host__ float3 operator*(const float3& a, float b) {
		return make_float3(a.x * b, a.y * b, a.z * b);
	}
	__device__ __host__ float3 operator/(const float3& a, float b) {
		return make_float3(a.x / b, a.y / b, a.z / b);
	}

	__device__ __host__ float dot(const float3& a, const float3& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__device__ __host__ float3 cross(const float3& a, const float3& b) {
		return make_float3((a).y * (b).z - (b).y * (a).z, (a).z * (b).x - (b).z * (a).x, (a).x * (b).y - (b).x * (a).y);
	}

	__device__ __host__ float length_squared(const float3& a, const float3& b) {
		return dot(a - b, a - b);
	}

	__device__ float length(const float3& a, const float3& b) {
		return __fsqrt_rn(dot(a - b, a - b));
	}

	//__host__ float length(const float3& a, const float3& b) {
	//	return sqrtf(dot(a - b, a - b));
	//}

	__device__ __host__ float length_squared(const float3& a) {
		return dot(a, a);
	}

	__device__ float length(const float3& a) {
		return __fsqrt_rn(dot(a, a));
	}

	//__host__ float length(const float3& a) {
	//	return sqrtf(dot(a, a));
	//}

	__device__ float3 normalize(const float3& a) {
		return a / length(a);
	}

	__device__ __host__ float distance_squared(const float3& a, const float3& b) {
		return dot((a - b), (a - b));
	}

	__device__ float distance(const float3& a, const float3& b) {
		return __fsqrt_rn(dot((a - b), (a - b)));
	}

	//__device__ __forceinline__
	//	float __dot(const float3& a, const float3& b)
	//{
	//	return a.x * b.x + a.y * b.y + a.z * b.z;
	//}

	//__device__ __forceinline__
	//	float3 __cross(const float3& a, const float3& b)
	//{
	//	return float3(a.y * b.z - b.y * a.z,
	//		a.z* b.x - b.z * a.x,
	//		a.x* b.y - b.x * a.y);
	//}

	//__device__ __forceinline__
	//	float __lengthSquared(const float3& v)
	//{
	//	return DOT(v, v);
	//}

	//__device__ __forceinline__
	//	float __length(const float3& v)
	//{
	//	return __fsqrt_rn(__lengthSquared(v));
	//}

	//__device__ __forceinline__
	//	float __distanceSquared(const float3& a, const float3& b)
	//{
	//	return __lengthSquared(a - b);
	//}

	//__device__ __forceinline__
	//	float __distance(const float3& a, const float3& b)
	//{
	//	return __fsqrt_rn(__distanceSquared(a, b));
	//}

	//__device__ __forceinline__
	//	float3 __normalize(const float3& v)
	//{
	//	return v / __length(v);
	//}

	__device__
		float pointTriangleDistance(const float3& point, const float3& v0, const float3& v1, const float3& v2) {
		// Calculate the normal of the triangle
		float3 normal = normalize(cross(v1 - v0, v2 - v0));

		// Calculate the distance between the point and the plane of the triangle
		float distance = dot(point - v0, normal);

		// Check if the point is above or below the triangle
		float3 projectedPoint = point - distance * normal;
		if (dot(normal, cross(v1 - v0, projectedPoint - v0)) > 0 &&
			dot(normal, cross(v2 - v1, projectedPoint - v1)) > 0 &&
			dot(normal, cross(v0 - v2, projectedPoint - v2)) > 0) {
			// Point is inside the triangle
			//return std::abs(distance);
			return distance;
		}
		else {
			// Point is outside the triangle
			// You can return the distance to the closest edge or vertex if needed
			return min(min(DISTANCE(point, v0), DISTANCE(point, v1)), DISTANCE(point, v2));
		}
	}

	struct FillFunctor
	{
		float* values;
		float3* positions;
		int countX;
		int countY;
		int countZ;
		float3 minPoint;
		float voxelSize;

		__device__
			void operator()(size_t index)
		{
			auto z = index / (countX * countY);
			auto y = (index % (countX * countY)) / countX;
			auto x = (index % (countX * countY)) % countX;

			float3 position = make_float3(
				minPoint.x + x * voxelSize + 0.5f * voxelSize,
				minPoint.y + y * voxelSize + 0.5f * voxelSize,
				minPoint.z + z * voxelSize + 0.5f * voxelSize);

			values[index] = -FLT_MAX;
			positions[index] = position;
		}
	};

	struct ApplyFunctor
	{
		float* values;
		float3* positions;
		int countX;
		int countY;
		int countZ;
		float3 minPoint;
		float voxelSize;

		float3* vertices;
		size_t verticesSize;
		GLuint* indices;
		size_t indicesSize;
		float4* colors;
		size_t colorsSize;

		__device__
			float operator()(const float3& position)//size_t index)
		{
			float distance = DISTANCE(position, make_float3(0.0f, 0.0f, 10.0f));

			//printf("distance : %f\n", distance);

			return distance;
		}
	};

	struct UpdateFunctor
	{
		float3 center;

		__device__
			float operator()(const float3& position)//size_t index)
		{
			float distance = DISTANCE(position, center);

			//printf("distance : %f\n", distance);

			if (2.5 < distance && distance < 3.0)
			{
				return distance;
			}
			else
			{
				return FLT_MAX;
			}
		}
	};

	TSDF::TSDF()
	{
	}

	TSDF::TSDF(float voxelSize, const float3& minPoint, const float3& maxPoint)
		: voxelSize(voxelSize)
	{
		auto xLength = maxPoint.x - minPoint.x;
		auto yLength = maxPoint.y - minPoint.y;
		auto zLength = maxPoint.z - minPoint.z;

		voxelCountX = (size_t)ceilf(xLength / voxelSize) + 1;
		voxelCountY = (size_t)ceilf(yLength / voxelSize) + 1;
		voxelCountZ = (size_t)ceilf(zLength / voxelSize) + 1;

		// Make voxel count even number
		if (voxelCountX % 2 == 1) voxelCountX++;
		if (voxelCountY % 2 == 1) voxelCountY++;
		if (voxelCountZ % 2 == 1) voxelCountZ++;

		values = thrust::device_vector<float>(voxelCountZ * voxelCountY * voxelCountX, FLT_MAX);
		positions = thrust::device_vector<float3>(voxelCountZ * voxelCountY * voxelCountX, make_float3(0.0f, 0.0f, 0.0f));

		centerPoint = (maxPoint + minPoint) * 0.5f;
		this->minPoint = make_float3(
			centerPoint.x - voxelSize * (voxelCountX / 2),
			centerPoint.y - voxelSize * (voxelCountY / 2),
			centerPoint.z - voxelSize * (voxelCountZ / 2));
		this->maxPoint = make_float3(
			centerPoint.x + voxelSize * (voxelCountX / 2),
			centerPoint.y + voxelSize * (voxelCountY / 2),
			centerPoint.z + voxelSize * (voxelCountZ / 2));

		nvtxRangePushA("@Arron/TSDF Fill");

		FillFunctor fillFunctor;
		fillFunctor.values = thrust::raw_pointer_cast(values.data());
		fillFunctor.positions = thrust::raw_pointer_cast(positions.data());
		fillFunctor.countX = voxelCountX;
		fillFunctor.countY = voxelCountY;
		fillFunctor.countZ = voxelCountZ;
		fillFunctor.minPoint = this->minPoint;
		fillFunctor.voxelSize = voxelSize;

		thrust::for_each(thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(values.size()), fillFunctor);

		nvtxRangePop();

		//printf("%d x %d x %d voxels\n", voxelCountX, voxelCountY, voxelCountZ);
	}

	void TSDF::Apply(Neon::Mesh* mesh)
	{
		auto casted_vertices = mesh->GetVertexBuffer()->Cast<float3>();
		thrust::device_vector<float3> vertices(
			casted_vertices.begin(),
			casted_vertices.end());

		thrust::device_vector<GLuint> indices(
			mesh->GetIndexBuffer()->GetElements().begin(),
			mesh->GetIndexBuffer()->GetElements().end());

		//thrust::device_vector<float4> colors(
		//	mesh->GetColorBuffer()->GetElements().begin(),
		//	mesh->GetColorBuffer()->GetElements().end());

		auto _values = thrust::raw_pointer_cast(values.data());
		auto _positions = thrust::raw_pointer_cast(positions.data());
		auto _vertices = thrust::raw_pointer_cast(vertices.data());
		auto _indices = thrust::raw_pointer_cast(indices.data());
		//auto _colors = thrust::raw_pointer_cast(colors.data());

		auto verticesSize = vertices.size();
		auto indicesSize = indices.size();
		//auto colorsSize = colors.size();

		auto _voxelSize = voxelSize;
		auto _voxelCountX = voxelCountX;
		auto _voxelCountY = voxelCountY;
		auto _voxelCountZ = voxelCountZ;

		auto voxelCount = voxelCountX * voxelCountY * voxelCountZ;

		auto _minPoint = minPoint;

		nvtxRangePushA("@Arron/Apply");

		ApplyFunctor applyFunctor;
		applyFunctor.values = _values;
		applyFunctor.positions = _positions;
		applyFunctor.countX = voxelCountX;
		applyFunctor.countY = voxelCountY;
		applyFunctor.countZ = voxelCountZ;
		applyFunctor.minPoint = this->minPoint;
		applyFunctor.voxelSize = voxelSize;
		applyFunctor.vertices = _vertices;
		applyFunctor.verticesSize = verticesSize;
		applyFunctor.indices = _indices;
		applyFunctor.indicesSize = indicesSize;
		//applyFunctor.colors = _colors;
		//applyFunctor.colorsSize = colorsSize;

		//thrust::for_each(thrust::make_counting_iterator<size_t>(0),
		//	thrust::make_counting_iterator<size_t>(values.size()), applyFunctor);

		thrust::transform(
			positions.begin(),
			positions.end(),
			values.begin(),
			applyFunctor);

		nvtxRangePop();
	}

	void TSDF::UpdateValues()
	{
		nvtxRangePushA("@Arron/UpdateValues");
		auto _values = thrust::raw_pointer_cast(values.data());
		auto _positions = thrust::raw_pointer_cast(positions.data());

		UpdateFunctor updateFunctor;
		updateFunctor.center = this->centerPoint;

		thrust::transform(
			positions.begin(),
			positions.end(),
			values.begin(),
			updateFunctor);

		nvtxRangePop();
	}

	void TSDF::Test(Neon::Scene* scene)
	{
		thrust::host_vector<float> host_values(values.begin(), values.end());

		int cnt = 0;

		for (size_t z = 0; z < voxelCountZ; z++)
		{
			char buffer[16]{ 0 };
			itoa(z, buffer, 10);

			auto debug = scene->Debug(buffer);

			for (size_t y = 0; y < voxelCountY; y++)
			{
				for (size_t x = 0; x < voxelCountX; x++)
				{
					float3 point = make_float3(
						minPoint.x + voxelSize * x,
						minPoint.y + voxelSize * y,
						minPoint.z + voxelSize * z);

					auto value = host_values[z * voxelCountY * voxelCountX + y * voxelCountX + x];

					if (cnt == 100)
					{
						if (value != FLT_MAX)
						{
							//debug->AddBox({ point.x, point.y, point.z }, voxelSize, voxelSize, voxelSize, glm::vec4((10.0f - value) / 10.0f, value / 10.0f, 1.0f, (10.0f - value) / 10.0f));
							debug->AddBox({ point.x, point.y, point.z }, voxelSize, voxelSize, voxelSize, glm::vec4((10.0f - value) / 10.0f, value / 10.0f, 1.0f, (10.0f - value) / 10.0f));
						}
						cnt = 0;
					}

					cnt++;
				}
			}
		}
	}
}