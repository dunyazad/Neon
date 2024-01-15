#include <Neon/CUDA/CUDASurfaceExtraction.h>

#include <Neon/NeonScene.h>
#include <Neon/NeonDebugEntity.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/NeonVertexBufferObject.hpp>

#undef min
#undef max

namespace NeonCUDA
{
	struct MinMaxFunctor {
		Eigen::Vector3f min_val;
		Eigen::Vector3f max_val;

		MinMaxFunctor() {
			min_val = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
			max_val = Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		}

		__host__ __device__
			void operator()(const Eigen::Vector3f& v) {
			if (VECTOR3_VALID(v))
			{
				min_val.x() = min(min_val.x(), v.x());
				min_val.y() = min(min_val.y(), v.y());
				min_val.z() = min(min_val.z(), v.z());

				max_val.x() = max(max_val.x(), v.x());
				max_val.y() = max(max_val.y(), v.y());
				max_val.z() = max(max_val.z(), v.z());
			}
		}
	};

	struct ExtractComponents : public thrust::unary_function<Eigen::Vector3f, float>
	{
		int component;

		ExtractComponents(int component) : component(component) {}

		__host__ __device__
			float operator()(const Eigen::Vector3f& v) const
		{
			return v(component);
		}
	};

	bool __device__ RayTriangleIntersect(const Eigen::Vector3f& ray_origin, const Eigen::Vector3f& ray_direction,
		const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, bool enable_backculling, float& distance)
	{
		using Eigen::Vector3f;
		const float epsilon = 1e-7f;

		const Vector3f v0v1 = v1 - v0;
		const Vector3f v0v2 = v2 - v0;

		const Vector3f pvec = ray_direction.cross(v0v2);

		const float det = v0v1.dot(pvec);

		if (enable_backculling)
		{
			// If det is negative, the triangle is back-facing.
			// If det is close to 0, the ray misses the triangle.
			if (det < epsilon)
				return false;
		}
		else
		{
			// If det is close to 0, the ray and triangle are parallel.
			if (std::abs(det) < epsilon)
				return false;
		}
		const float inv_det = 1 / det;

		const Vector3f tvec = ray_origin - v0;
		const auto u = tvec.dot(pvec) * inv_det;
		if (u < 0 || u > 1)
			return false;

		const Vector3f qvec = tvec.cross(v0v1);
		const auto v = ray_direction.dot(qvec) * inv_det;
		if (v < 0 || u + v > 1)
			return false;

		const auto t = v0v2.dot(qvec) * inv_det;

		distance = t;
		return true;
	}

	void DoSurfaceExtractionWrapper(Neon::Scene* scene, Neon::Mesh* mesh, const Eigen::Matrix4f& transform)
	{
		size_t hResolution = 256;
		size_t vResolution = 480;
		float xUnit = 0.1f;
		float yUnit = 0.1f;
		float voxelSize = 0.1f;

		auto inputPositions = mesh->GetVertexBuffer()->GetElements();

		Eigen::AlignedBox3f aabb;

#pragma region Mesh Vertices
		thrust::host_vector<Eigen::Vector3f> host_meshVertices;
		for (auto& p : inputPositions)
		{
			auto v = Eigen::Vector3f(p.x, p.y, p.z);
			host_meshVertices.push_back(v);
			if (VECTOR3_VALID(v))
			{
				aabb.extend(v);
			}

			scene->Debug("Point Input SurfaceExtraction")->AddPoint({ p.x, p.y, 0.0f });
		}
		thrust::device_vector<Eigen::Vector3f> inputPoints(host_meshVertices.begin(), host_meshVertices.end());
#pragma endregion

		DoSurfaceExtraction(scene, inputPoints, aabb, transform);
	}

	void DoSurfaceExtraction(Neon::Scene* scene, const thrust::device_vector<Eigen::Vector3f>& inputPoints, const Eigen::AlignedBox3f& aabb, const Eigen::Matrix4f& transform)
	{
		size_t hResolution = 256;
		size_t vResolution = 480;
		float xUnit = 0.1f;
		float yUnit = 0.1f;
		float voxelSize = 0.1f;

#pragma region Get transformed AABB of inputPoints
		auto transformedAABB = aabb;
		transformedAABB.transform(Eigen::Transform<float, 3, Eigen::Affine>(transform));
		Neon::AABB aabbaabb({ transformedAABB.min().x(), transformedAABB.min().y(), transformedAABB.min().z() }, { transformedAABB.max().x(), transformedAABB.max().y(), transformedAABB.max().z() });
		scene->Debug("aabb")->AddAABB(aabbaabb);
#pragma endregion


#pragma region Mesh Indices
		nvtxRangePushA("@Aaron/Build Mesh Indices");
		thrust::device_vector<size_t> meshIndices(hResolution * vResolution * 6);
		auto _meshIndices = thrust::raw_pointer_cast(meshIndices.data());

		thrust::for_each(
			thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>((hResolution - 1) * (vResolution - 1)),
			[_meshIndices, hResolution, vResolution]__device__(size_t index) {

			auto y = index / hResolution;
			auto x = index % hResolution;

			if (0 == x % 2 && 0 == y % 3)
			{
				auto i0 = hResolution * y + x;
				auto i1 = hResolution * y + x + 2;
				auto i2 = hResolution * (y + 3) + x;
				auto i3 = hResolution * (y + 3) + x + 2;

				if ((i0 >= hResolution * vResolution) ||
					(i1 >= hResolution * vResolution) ||
					(i2 >= hResolution * vResolution) ||
					(i3 >= hResolution * vResolution))
					return;

				_meshIndices[index * 6 + 0] = i0;
				_meshIndices[index * 6 + 1] = i1;
				_meshIndices[index * 6 + 2] = i2;

				_meshIndices[index * 6 + 3] = i2;
				_meshIndices[index * 6 + 4] = i1;
				_meshIndices[index * 6 + 5] = i3;
			}
		});
		nvtxRangePop();
#pragma endregion


	}

	SurfaceExtractor::SurfaceExtractor(size_t hResolution, size_t vResolution, float voxelSize)
		: hResolution(hResolution), vResolution(vResolution), voxelSize(voxelSize)
	{
		Initialize();
	}

	SurfaceExtractor::~SurfaceExtractor()
	{
	}

	void SurfaceExtractor::Initialize()
	{
		voxelValues = thrust::device_vector<float>(hResolution * vResolution * hResolution, FLT_MAX);
		//auto _voxelValues = thrust::raw_pointer_cast(voxelValues.data());

		voxelCenterPositions = thrust::device_vector<Eigen::Vector3f>(hResolution * vResolution * hResolution, Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));
		//auto _voxelCenterPositions = thrust::raw_pointer_cast(voxelCenterPositions.data());

		nvtxRangePushA("@Aaron/Build Mesh Indices");
		meshIndices = thrust::device_vector<GLuint>(hResolution * vResolution * 6);
		auto _meshIndices = thrust::raw_pointer_cast(meshIndices.data());

		GLuint _hResolution = (GLuint)hResolution;
		GLuint _vResolution = (GLuint)vResolution;

		thrust::for_each(
			thrust::make_counting_iterator<GLuint>(0),
			thrust::make_counting_iterator<GLuint>((hResolution - 1) * (vResolution - 1)),
			[_meshIndices, _hResolution, _vResolution]__device__(GLuint index) {

			GLuint y = index / _hResolution;
			GLuint x = index % _hResolution;

			if (0 == x % 2 && 0 == y % 3)
			{
				GLuint i0 = _hResolution * y + x;
				GLuint i1 = _hResolution * y + x + 2;
				GLuint i2 = _hResolution * (y + 3) + x;
				GLuint i3 = _hResolution * (y + 3) + x + 2;

				if ((i0 >= _hResolution * _vResolution) ||
					(i1 >= _hResolution * _vResolution) ||
					(i2 >= _hResolution * _vResolution) ||
					(i3 >= _hResolution * _vResolution))
					return;

				_meshIndices[index * 6 + 0] = i0;
				_meshIndices[index * 6 + 1] = i1;
				_meshIndices[index * 6 + 2] = i2;

				_meshIndices[index * 6 + 3] = i2;
				_meshIndices[index * 6 + 4] = i1;
				_meshIndices[index * 6 + 5] = i3;
			}
		});
		nvtxRangePop();
	}

	void SurfaceExtractor::PrepareNewFrame()
	{
		nvtxRangePushA("@Aaron/SurfaceExtractor::PrepareNewFrame()");

		thrust::fill(voxelValues.begin(), voxelValues.end(), FLT_MAX);
		thrust::fill(voxelCenterPositions.begin(), voxelCenterPositions.end(), Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));
		lastFrameAABB.setEmpty();

		nvtxRangePop();
	}

	void SurfaceExtractor::NewFrameWrapper(Neon::Scene* scene, Neon::Mesh* mesh, const Eigen::Matrix4f& transform)
	{
		this->scene = scene;

		Eigen::AlignedBox3f aabb;

		thrust::host_vector<Eigen::Vector3f> host_meshVertices;
		for (auto& p : mesh->GetVertexBuffer()->GetElements())
		{
			auto v = Eigen::Vector3f(p.x, p.y, p.z);
			host_meshVertices.push_back(v);
			if (VECTOR3_VALID(v))
			{
				aabb.extend(v);
			}
		}
		thrust::device_vector<Eigen::Vector3f> inputPoints(host_meshVertices.begin(), host_meshVertices.end());

#pragma region Draw Trinagles using Input Points
		{
			for (size_t y = 0; y < 480 - 3; y += 3)
			{
				for (size_t x = 0; x < 256 - 2; x += 2)
				{
					auto i0 = 256 * y + x;
					auto i1 = 256 * y + x + 2;
					auto i2 = 256 * (y + 3) + x;
					auto i3 = 256 * (y + 3) + x + 2;

					auto& p0 = host_meshVertices[i0];
					auto& p1 = host_meshVertices[i1];
					auto& p2 = host_meshVertices[i2];
					auto& p3 = host_meshVertices[i3];

					auto v0 = transform * Eigen::Vector4f(p0.x(), p0.y(), p0.z(), 1.0f);
					auto v1 = transform * Eigen::Vector4f(p1.x(), p1.y(), p1.z(), 1.0f);
					auto v2 = transform * Eigen::Vector4f(p2.x(), p2.y(), p2.z(), 1.0f);
					auto v3 = transform * Eigen::Vector4f(p3.x(), p3.y(), p3.z(), 1.0f);

					scene->Debug("input triangles1")->AddTriangle(
						{ v0.x(), v0.y(), v0.z() },
						{ v1.x(), v1.y(), v1.z() },
						{ v2.x(), v2.y(), v2.z() },
						glm::green, glm::green, glm::green);

					scene->Debug("input triangles1")->AddTriangle(
						{ v2.x(), v2.y(), v2.z() },
						{ v1.x(), v1.y(), v1.z() },
						{ v3.x(), v3.y(), v3.z() },
						glm::green, glm::green, glm::green);
				}
			}
		}
#pragma endregion

#pragma region Draw Triangles using meshIndices
		//{
		//	auto host_meshIndices = thrust::host_vector<GLuint>(meshIndices);
		//	for (GLuint i = 0; i < (GLuint)hResolution * (GLuint)vResolution * 2; i++)
		//	{
		//		GLuint i0 = host_meshIndices[i * 3 + 0];
		//		GLuint i1 = host_meshIndices[i * 3 + 1];
		//		GLuint i2 = host_meshIndices[i * 3 + 2];

		//		auto& p0 = host_meshVertices[i0];
		//		auto& p1 = host_meshVertices[i1];
		//		auto& p2 = host_meshVertices[i2];

		//		auto v0 = transform * Eigen::Vector4f(p0.x(), p0.y(), p0.z(), 1.0f);
		//		auto v1 = transform * Eigen::Vector4f(p1.x(), p1.y(), p1.z(), 1.0f);
		//		auto v2 = transform * Eigen::Vector4f(p2.x(), p2.y(), p2.z(), 1.0f);

		//		if (false == VECTOR3_VALID(v0) || false == VECTOR3_VALID(v1) || false == VECTOR3_VALID(v2))
		//			continue;

		//		scene->Debug("input triangles")->AddTriangle(
		//			{ v0.x(), v0.y(), v0.z() },
		//			{ v2.x(), v2.y(), v2.z() },
		//			{ v1.x(), v1.y(), v1.z() },
		//			glm::green, glm::green, glm::green);
		//	}
		//}
#pragma endregion

		NewFrame(inputPoints, aabb, transform);
	}

	void SurfaceExtractor::NewFrame(const thrust::device_vector<Eigen::Vector3f>& inputPoints, const Eigen::AlignedBox3f& aabb, const Eigen::Matrix4f& transform)
	{
		nvtxRangePushA("@Aaron/SurfaceExtractor::NewFrame()");

		PrepareNewFrame();

		auto transformedAABB = aabb;
		transformedAABB.transform(Eigen::Transform<float, 3, Eigen::Affine>(transform));

		float xmin = floorf(transformedAABB.min().x() / voxelSize) * voxelSize;
		float ymin = floorf(transformedAABB.min().y() / voxelSize) * voxelSize;
		float zmin = floorf(transformedAABB.min().z() / voxelSize) * voxelSize;

		float xmax = ceilf(transformedAABB.max().x() / voxelSize) * voxelSize;
		float ymax = ceilf(transformedAABB.max().y() / voxelSize) * voxelSize;
		float zmax = ceilf(transformedAABB.max().z() / voxelSize) * voxelSize;

		lastFrameAABB = Eigen::AlignedBox3f(Eigen::Vector3f(xmin, ymin, zmin), Eigen::Vector3f(xmax, ymax, zmax));
		voxelCountX = (size_t)((xmax - xmin) / voxelSize);
		voxelCountY = (size_t)((ymax - ymin) / voxelSize);
		voxelCountZ = (size_t)((zmax - zmin) / voxelSize);

		auto _lastFrameAABB = lastFrameAABB;
		auto _voxelCountX = voxelCountX;
		auto _voxelCountY = voxelCountY;
		auto _voxelCountZ = voxelCountZ;
		auto _voxelSize = voxelSize;

		auto _voxelCenterPositions = thrust::raw_pointer_cast(voxelCenterPositions.data());

		thrust::for_each(
			thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(hResolution * vResolution * hResolution),
			[_lastFrameAABB, _voxelCountX, _voxelCountY, _voxelCountZ, _voxelSize, _voxelCenterPositions]
			__device__(size_t index) {

			auto zIndex = index / (_voxelCountX * _voxelCountY);
			auto yIndex = (index % (_voxelCountX * _voxelCountY)) / _voxelCountX;
			auto xIndex = (index % (_voxelCountX * _voxelCountY)) % _voxelCountX;

			float xpos = _lastFrameAABB.min().x() + xIndex * _voxelSize + _voxelSize * 0.5f;
			float ypos = _lastFrameAABB.min().y() + yIndex * _voxelSize + _voxelSize * 0.5f;
			float zpos = _lastFrameAABB.min().z() + zIndex * _voxelSize + _voxelSize * 0.5f;

			_voxelCenterPositions[index].x() = xpos;
			_voxelCenterPositions[index].y() = ypos;
			_voxelCenterPositions[index].z() = zpos;
		});

#pragma region Visualize Voxel Center Positions
		//{
		//	auto host_voxelCenterPositions = thrust::host_vector<Eigen::Vector3f>(voxelCenterPositions);

		//	for (size_t i = 0; i < host_voxelCenterPositions.size(); i++)
		//	{
		//		auto& v = host_voxelCenterPositions[i];

		//		if (lastFrameAABB.contains(v))
		//		{
		//			scene->Debug("Voxel Positions")->AddPoint(glm::make_vec3(v.data()), glm::red);
		//		}
		//	}
		//}
#pragma endregion

		auto _inputPoints = thrust::raw_pointer_cast(inputPoints.data());
		auto _meshIndices = thrust::raw_pointer_cast(meshIndices.data());
		auto _voxelValues = thrust::raw_pointer_cast(voxelValues.data());
		auto _transform = transform;

		thrust::for_each(
			thrust::make_counting_iterator<GLuint>(0),
			thrust::make_counting_iterator<GLuint>(hResolution * vResolution * 2),
			[_inputPoints, _meshIndices, _transform, _voxelValues, _lastFrameAABB, _voxelCountX, _voxelCountY, _voxelCountZ, _voxelSize]
			__device__(GLuint index) {
			GLuint i0 = _meshIndices[index * 3 + 0];
			GLuint i1 = _meshIndices[index * 3 + 1];
			GLuint i2 = _meshIndices[index * 3 + 2];

			auto& p0 = _inputPoints[i0];
			auto& p1 = _inputPoints[i1];
			auto& p2 = _inputPoints[i2];

			auto tv0 = _transform * Eigen::Vector4f(p0.x(), p0.y(), p0.z(), 1.0f);
			auto tv1 = _transform * Eigen::Vector4f(p1.x(), p1.y(), p1.z(), 1.0f);
			auto tv2 = _transform * Eigen::Vector4f(p2.x(), p2.y(), p2.z(), 1.0f);

			auto v0 = Eigen::Vector3f(tv0.x(), tv0.y(), tv0.z());
			auto v1 = Eigen::Vector3f(tv1.x(), tv1.y(), tv1.z());
			auto v2 = Eigen::Vector3f(tv2.x(), tv2.y(), tv2.z());

			if (false == VECTOR3_VALID(v0) || false == VECTOR3_VALID(v1) || false == VECTOR3_VALID(v2))
				return;

			auto aabb = Eigen::AlignedBox3f();
			aabb.extend(v0);
			aabb.extend(v1);
			aabb.extend(v2);

			size_t xminIndex = (size_t)floorf((aabb.min().x() - _lastFrameAABB.min().x()) / _voxelSize);
			size_t yminIndex = (size_t)floorf((aabb.min().y() - _lastFrameAABB.min().y()) / _voxelSize);
			size_t zminIndex = (size_t)floorf((aabb.min().z() - _lastFrameAABB.min().z()) / _voxelSize);

			size_t xmaxIndex = (size_t)ceilf((aabb.max().x() - _lastFrameAABB.min().x()) / _voxelSize);
			size_t ymaxIndex = (size_t)ceilf((aabb.max().y() - _lastFrameAABB.min().y()) / _voxelSize);
			size_t zmaxIndex = (size_t)ceilf((aabb.max().z() - _lastFrameAABB.min().z()) / _voxelSize);

			for (size_t z = zminIndex; z < zmaxIndex; z++)
			{
				for (size_t y = yminIndex; y < ymaxIndex; y++)
				{
					for (size_t x = xminIndex; x < xmaxIndex; x++)
					{
						//_voxelValues[z * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x] = 100.0f;

						float xpos = _lastFrameAABB.min().x() + x * _voxelSize + _voxelSize * 0.5f;
						float ypos = _lastFrameAABB.min().y() + y * _voxelSize + _voxelSize * 0.5f;
						float zpos = _lastFrameAABB.min().z() + z * _voxelSize + _voxelSize * 0.5f;

						float distance = FLT_MAX;
						if (RayTriangleIntersect(Eigen::Vector3f(xpos, ypos, zpos), Eigen::Vector3f(0.0f, 0.0f, 1.0f),
							v0, v1, v2, false, distance))
						{
							_voxelValues[z * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x] = distance;

							for (size_t i = 0; i < 5; i++)
							{
								if (z - i >= 0)
								{
									_voxelValues[(z - i) * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x] = distance - i * _voxelSize;
								}

								if (z + i <= _voxelCountZ)
								{
									_voxelValues[(z + i) * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x] = distance + i * _voxelSize;
								}
							}
						}
					}
				}
			}
		});

		auto host_voxelValues = thrust::host_vector<float>(voxelValues);
		for (size_t i = 0; i < hResolution * vResolution * hResolution; i++)
		{
			float distance = host_voxelValues[i];
			if (FLT_VALID(distance))
			{
				auto zIndex = i / (_voxelCountX * _voxelCountY);
				auto yIndex = (i % (_voxelCountX * _voxelCountY)) / _voxelCountX;
				auto xIndex = (i % (_voxelCountX * _voxelCountY)) % _voxelCountX;

				float xpos = lastFrameAABB.min().x() + xIndex * voxelSize + voxelSize * 0.5f;
				float ypos = lastFrameAABB.min().y() + yIndex * voxelSize + voxelSize * 0.5f;
				float zpos = lastFrameAABB.min().z() + zIndex * voxelSize + voxelSize * 0.5f;

				if (distance < -1.0f) distance = -1.0f;
				if (distance > 1.0f) distance = 1.0f;

				float ratio = (distance + 1.0f) / 2.0f;

				glm::vec4 c = (1.0f - ratio) * glm::blue + ratio * glm::red;
				c.a = 1.0f;

				scene->Debug("Mesh Contained Voxels")->AddPoint({ xpos, ypos, zpos }, c);
			}
		}

		nvtxRangePop();
	}
}
