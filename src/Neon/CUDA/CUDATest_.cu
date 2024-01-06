#include <Neon/CUDA/CUDATest.h>

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonMesh.h>

#include <Neon/CUDA/CUDACommon.h>
//#include <nvtx3/nvToolsExt.h>




namespace NeonCUDA
{
	__device__ void print(Eigen::Vector3f* input, size_t inputSize)
	{
		for (size_t i = 0; i < inputSize; i++)
		{
			auto v = input[i];

			printf("%f, %f, %f\n", v.x(), v.y(), v.z());
		}
	}

	__global__ void WrapPrint(Eigen::Vector3f* input, size_t inputSize)
	{
		print(input, inputSize);
	}

	__device__
		bool almost_equal(float a, float b)
	{
		return fabsf(a - b) < CUDA_EPSILON;
	}

	__device__
		bool almost_equal(const Eigen::Vector2f& a, const Eigen::Vector2f& b)
	{
		return almost_equal(a.x(), b.x()) &&
			almost_equal(a.y(), b.y());
	}

	__device__
		bool almost_equal(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
	{
		return almost_equal(a.x(), b.x()) &&
			almost_equal(a.y(), b.y()) &&
			almost_equal(a.z(), b.z());
	}

	__device__
		bool Edge::operator == (const Edge& t)
	{
		return (v0 == t.v0 || v0 == t.v1) && (v1 == t.v0 || v1 == t.v1);
	}

	__device__ __host__
		bool almost_equal(const Edge& e0, const Edge& e1)
	{
		return	(almost_equal(e0.points[e0.v0], e1.points[e1.v0]) && almost_equal(e0.points[e0.v1], e1.points[e1.v1])) ||
			(almost_equal(e0.points[e0.v0], e1.points[e1.v1]) && almost_equal(e0.points[e0.v1], e1.points[e1.v0]));
	}

	__device__
		bool Triangle::containsVertex(VertexIndex v)
	{
		return almost_equal(points[v0], points[v]) ||
			almost_equal(points[v1], points[v]) ||
			almost_equal(points[v2], points[v]);
	}

	__device__
	bool Triangle::aabbContains(VertexIndex v)
	{
		auto x0 = points[v0].x();
		auto y0 = points[v0].y();

		auto x1 = points[v1].x();
		auto y1 = points[v1].y();

		auto x2 = points[v2].x();
		auto y2 = points[v2].y();

		auto minX = x0; if (minX > x1) minX = x1; if (minX > x2) minX = x2;
		auto minY = y0; if (minY > y1) minY = y1; if (minY > y2) minY = y2;

		auto maxX = x0; if (maxX < x1) maxX = x1; if (maxX < x2) maxX = x2;
		auto maxY = y0; if (maxY < y1) maxY = y1; if (maxY < y2) maxY = y2;

		auto& p = points[v];
		return (minX <= p.x() && p.x() <= maxX) && (minY <= p.y() && p.y() <= maxY);
	}

	__device__
		bool Triangle::circumCircleContains(VertexIndex v)
	{
		auto sn0 = points[v0].squaredNorm();
		auto sn1 = points[v1].squaredNorm();
		auto sn2 = points[v2].squaredNorm();

		auto x0 = points[v0].x();
		auto y0 = points[v0].y();

		auto x1 = points[v1].x();
		auto y1 = points[v1].y();

		auto x2 = points[v2].x();
		auto y2 = points[v2].y();

		auto circumX = (sn0 * (y2 - y1) + sn1 * (y0 - y2) + sn2 * (y1 - y0)) / (x0 * (y2 - y1) + x1 * (y0 - y2) + x2 * (y1 - y0));
		auto circumY = (sn0 * (x2 - x1) + sn1 * (x0 - x2) + sn2 * (x1 - x0)) / (y0 * (x2 - x1) + y1 * (x0 - x2) + y2 * (x1 - x0));

		Eigen::Vector2f circum(circumX * 0.5f, circumY * 0.5f);
		auto circumRadius = __fsqrt_rn((circum - Eigen::Vector2f(x0, y0)).squaredNorm());
		auto dist = __fsqrt_rn((circum - Eigen::Vector2f(points[v].x(), points[v].y())).squaredNorm());
		//printf("[%d, %d, %d] dist: %f, circumRadius: %f\n", v0, v1, v2, dist, circumRadius);
		return dist <= circumRadius;
	}

	__device__
		bool Triangle::operator == (const Triangle& t)
	{
		return (v0 == t.v0 || v0 == t.v1 || v0 == t.v2) &&
			(v1 == t.v0 || v1 == t.v1 || v1 == t.v2) &&
			(v2 == t.v0 || v2 == t.v1 || v2 == t.v2);
	}

	__device__ void GetSuperTriangle(Eigen::Vector3f* points, size_t pointsSize, Triangle* triangles, int* trianglesSize)
	{
		float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
		float maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;

		for (size_t i = 0; i < pointsSize - 3; i++)
		{
			auto& v = points[i];

			if (minX > v.x()) minX = v.x();
			if (minY > v.y()) minY = v.y();
			if (minZ > v.z()) minZ = v.z();

			if (maxX < v.x()) maxX = v.x();
			if (maxY < v.y()) maxY = v.y();
			if (maxZ < v.z()) maxZ = v.z();
		}

		float centerX = (minX + maxX) * 0.5f;
		float centerY = (minY + maxY) * 0.5f;
		float centerZ = (minZ + maxZ) * 0.5f;

		points[pointsSize - 3 + 0].x() = minX + (minX - centerX) * 3;
		points[pointsSize - 3 + 0].y() = minY + (minY - centerY) * 3;
		points[pointsSize - 3 + 0].z() = 0.0f;

		points[pointsSize - 3 + 1].x() = centerX;
		points[pointsSize - 3 + 1].y() = maxY + (maxY - centerY) * 3;
		points[pointsSize - 3 + 1].z() = 0.0f;

		points[pointsSize - 3 + 2].x() = maxX + (maxX - centerX) * 3;
		points[pointsSize - 3 + 2].y() = minY + (minY - centerY) * 3;
		points[pointsSize - 3 + 2].z() = 0.0f;

		trianglesSize[0] = 1;

		triangles[0].points = points;
		triangles[0].v0 = pointsSize - 3 + 0;
		triangles[0].v1 = pointsSize - 3 + 1;
		triangles[0].v2 = pointsSize - 3 + 2;
	}

	__global__ void GetSuperTriangleWrapper(Eigen::Vector3f* points, size_t pointsSize, Triangle* triangles, int* trianglesSize)
	{
		GetSuperTriangle(points, pointsSize, triangles, trianglesSize);

		//printf("trianglesSize : %d\n", trianglesSize);
	}

	thrust::host_vector<Eigen::Vector3i> DelaunayTriangulation_BowyerWatson(vector<Eigen::Vector3f>& inputPoints)
	{
		thrust::device_vector<Eigen::Vector3f> device_points(inputPoints.size() + 3);
		thrust::copy(inputPoints.begin(), inputPoints.end(), device_points.begin());
		
		vector<Eigen::Vector3i> result;
		
		thrust::device_vector<Triangle> triangles((inputPoints.size() + 2) * 3);
		thrust::device_vector<int> trianglesSize(1);

		auto _points = thrust::raw_pointer_cast(device_points.data());
		auto _pointsSize = inputPoints.size();
		auto _triangles = thrust::raw_pointer_cast(triangles.data());
		auto _trianglesSize = thrust::raw_pointer_cast(trianglesSize.data());

		thrust::device_vector<Edge> edges(inputPoints.size() * 100);
		thrust::device_vector<int> edgesSizes(inputPoints.size(), 0);

		auto _edges = thrust::raw_pointer_cast(edges.data());
		auto _edgesSizes = thrust::raw_pointer_cast(edgesSizes.data());

		nvtxRangePushA("@Aaron/DT");
		nvtxRangePushA("@Aaron/DT/GetSuperTriangle");

		GetSuperTriangleWrapper<<<1, 1>>>(
			thrust::raw_pointer_cast(device_points.data()), device_points.size(),
			thrust::raw_pointer_cast(triangles.data()), thrust::raw_pointer_cast(trianglesSize.data()));
		nvtxRangePop();

		nvtxRangePushA("@Aaron/DT/Main");
		thrust::for_each(
			thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(1),
			[_points, _pointsSize, _edges, _edgesSizes, _triangles, _trianglesSize] __device__(size_t not_using) {
			for (size_t index = 0; index < _pointsSize; index++)
			{
				for (size_t i = 0; i < _trianglesSize[0]; i++)
				{
					_triangles[i].points = _points;

					if (_triangles[i].aabbContains(index))
					{
						if (_triangles[i].circumCircleContains(index))
						{
							_triangles[i].isBad = true;
							(_edges[index * 100 + _edgesSizes[index]]).isBad = false;
							(_edges[index * 100 + _edgesSizes[index]]).points = _points;
							(_edges[index * 100 + _edgesSizes[index]]).v0 = _triangles[i].v0;
							(_edges[index * 100 + _edgesSizes[index]]).v1 = _triangles[i].v1;
							_edgesSizes[index]++;

							(_edges[index * 100 + _edgesSizes[index]]).isBad = false;
							(_edges[index * 100 + _edgesSizes[index]]).points = _points;
							(_edges[index * 100 + _edgesSizes[index]]).v0 = _triangles[i].v1;
							(_edges[index * 100 + _edgesSizes[index]]).v1 = _triangles[i].v2;
							_edgesSizes[index]++;

							(_edges[index * 100 + _edgesSizes[index]]).isBad = false;
							(_edges[index * 100 + _edgesSizes[index]]).points = _points;
							(_edges[index * 100 + _edgesSizes[index]]).v0 = _triangles[i].v2;
							(_edges[index * 100 + _edgesSizes[index]]).v1 = _triangles[i].v0;
							_edgesSizes[index]++;
						}
					}
				}

				size_t squeezedTriangleIndex = 0;
				for (size_t i = 0; i < _trianglesSize[0]; i++)
				{
					auto& t = _triangles[i];
					if (t.isBad == false)
					{
						_triangles[squeezedTriangleIndex].isBad = t.isBad;
						_triangles[squeezedTriangleIndex].points = t.points;
						_triangles[squeezedTriangleIndex].v0 = t.v0;
						_triangles[squeezedTriangleIndex].v1 = t.v1;
						_triangles[squeezedTriangleIndex].v2 = t.v2;
						squeezedTriangleIndex++;
					}
					else
					{
						t.isBad = true;
						t.points = nullptr;
						t.v0 = -1;
						t.v1 = -1;
						t.v2 = -1;
					}
				}
				_trianglesSize[0] = squeezedTriangleIndex;

				for (size_t i = 0; i < _edgesSizes[index]; i++)
				{
					for (size_t j = i + 1; j < _edgesSizes[index]; j++)
					{
						if (almost_equal(_edges[index * 100 + i], _edges[index * 100 + j]))
						{
							_edges[index * 100 + i].isBad = true;
							_edges[index * 100 + j].isBad = true;
						}
					}
				}

				size_t squeezedEdgeIndex = 0;
				for (size_t i = 0; i < _edgesSizes[index]; i++)
				{
					auto& edge = _edges[index * 100 + i];
					if (edge.isBad == false)
					{
						_edges[index * 100 + squeezedEdgeIndex].isBad = edge.isBad;
						_edges[index * 100 + squeezedEdgeIndex].points = edge.points;
						_edges[index * 100 + squeezedEdgeIndex].v0 = edge.v0;
						_edges[index * 100 + squeezedEdgeIndex].v1 = edge.v1;
						squeezedEdgeIndex++;
					}
					else
					{
						edge.isBad = true;
						edge.points = nullptr;
						edge.v0 = -1;
						edge.v1 = -1;
					}
				}
				_edgesSizes[index] = squeezedEdgeIndex;

				for (size_t i = 0; i < _edgesSizes[index]; i++)
				{
					auto& edge = _edges[index * 100 + i];

					_triangles[_trianglesSize[0]].isBad = false;
					_triangles[_trianglesSize[0]].points = _points;
					_triangles[_trianglesSize[0]].v0 = edge.v0;
					_triangles[_trianglesSize[0]].v1 = edge.v1;
					_triangles[_trianglesSize[0]].v2 = index;
					_trianglesSize[0]++;
				}

				for (size_t i = 0; i < _edgesSizes[index]; i++)
				{
					auto& edge = _edges[index * 100 + i];
					edge.isBad = true;
					edge.points = nullptr;
					edge.v0 = -1;
					edge.v1 = -1;
				}
				_edgesSizes[index] = 0;
			}
		});
		nvtxRangePop();

		thrust::host_vector<Triangle> temp_triangles(triangles.begin(), triangles.end());
		for (auto& t : temp_triangles)
		{
			result.push_back(Eigen::Vector3i(t.v0, t.v1, t.v2));
		}


		//nvtxRangePushA("@Aaron/DT/for_each");
		//thrust::for_each(
		//	thrust::make_counting_iterator<size_t>(0),
		//	thrust::make_counting_iterator<size_t>(inputPoints.size()),
		//	[_points, _edges, _edgesSizes, _triangles, _trianglesSize]__device__(size_t index) {

		//	for (size_t i = 0; i < _trianglesSize[0]; i++)
		//	{
		//		_triangles[i].points = _points;
		//		if (_triangles[i].circumCircleContains(index))
		//		{
		//			_triangles[i].isBad = true;
		//			(_edges[index * 100 + _edgesSizes[index]]).points = _points;
		//			(_edges[index * 100 + _edgesSizes[index]]).v0 = _triangles[i].v0;
		//			(_edges[index * 100 + _edgesSizes[index]]).v1 = _triangles[i].v1;
		//			_edgesSizes[index]++;

		//			(_edges[index * 100 + _edgesSizes[index]]).points = _points;
		//			(_edges[index * 100 + _edgesSizes[index]]).v0 = _triangles[i].v1;
		//			(_edges[index * 100 + _edgesSizes[index]]).v1 = _triangles[i].v2;
		//			_edgesSizes[index]++;

		//			(_edges[index * 100 + _edgesSizes[index]]).points = _points;
		//			(_edges[index * 100 + _edgesSizes[index]]).v0 = _triangles[i].v2;
		//			(_edges[index * 100 + _edgesSizes[index]]).v1 = _triangles[i].v0;
		//			_edgesSizes[index]++;
		//		}
		//	}
		//});
		//nvtxRangePop();

		//nvtxRangePushA("@Aaron/DT/squeeze");
		//auto newEnd = thrust::copy_if(triangles.begin(), triangles.end(), triangles.begin(), []__device__(const Triangle& t) {
		//	return t.isBad == false;
		//});

		//auto newTrianglesSize = thrust::distance(triangles.begin(), newEnd);
		//thrust::for_each(
		//	thrust::make_counting_iterator<size_t>(0),
		//	thrust::make_counting_iterator<size_t>(1),
		//	[newTrianglesSize, _trianglesSize]__device__(size_t index) {
		//	_trianglesSize[0] = newTrianglesSize;
		//});
		//nvtxRangePop();

		//nvtxRangePushA("@Aaron/DT/squeeze edges");
		//thrust::for_each(
		//	thrust::make_counting_iterator<size_t>(0),
		//	thrust::make_counting_iterator<size_t>(inputPoints.size()),
		//	[_points, _edges, _edgesSizes, _triangles, _trianglesSize]__device__(size_t index) {
		//	for (size_t i = 0; i < _edgesSizes[index]; i++)
		//	{
		//		for (size_t j = 1; j < _edgesSizes[index]; j++)
		//		{
		//			if (almost_equal(_edges[index * 100 + i], _edges[index * 100 + j]))
		//			{
		//				_edges[index * 100 + i].isBad = true;
		//				_edges[index * 100 + j].isBad = true;
		//			}
		//		}
		//	}
		//});
		//nvtxRangePop();

		//printf("New Trinagles Size: %d\n", newTrianglesSize);


		//PrintEdgesWrapper<<<1, 1>>>(thrust::raw_pointer_cast(edges.data()), inputPoints.size());

		//thrust::host_vector<Eigen::Vector3f> host_points(inputPoints.size() + 3);
		//thrust::copy_n(device_points.begin(), inputPoints.size() + 3, host_points.begin());
		//for (size_t i = 0; i < host_points.size(); i++)
		//{
		//	auto& v = host_points[i];
		//	printf("hp [%d] %f %f %f\n", i, v.x(), v.y(), v.z());
		//}

		//thrust::host_vector<int> host_trianglesSize(trianglesSize.begin(), trianglesSize.end());
		//printf("Triangles Size : %d\n", host_trianglesSize[0]);

		//thrust::host_vector<Triangle> host_triangles(triangles.begin(), triangles.end());
		//for (size_t i = 0; i < host_trianglesSize[0]; i++)
		//{
		//	auto& t = host_triangles[i];
		//	printf("Triangle : %d, %d, %d\n", t.v0, t.v1, t.v2);
		//}

		nvtxRangePop();

		return result;
	}
}
