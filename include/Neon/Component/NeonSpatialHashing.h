#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Mesh;

	template<typename V, typename T>
	class SpatialHashingCell : public AABB {
	public:
		SpatialHashingCell(const glm::vec3& minPoint = { FLT_MAX, FLT_MAX, FLT_MAX },
			const glm::vec3& maxPoint = { -FLT_MAX, -FLT_MAX, -FLT_MAX })
			: AABB(minPoint, maxPoint), index({ IntInfinity, IntInfinity, IntInfinity }) {}

		SpatialHashingCell(tuple<int, int, int> index,
			const glm::vec3& minPoint = { FLT_MAX, FLT_MAX, FLT_MAX },
			const glm::vec3& maxPoint = { -FLT_MAX, -FLT_MAX, -FLT_MAX })
			: AABB(minPoint, maxPoint), index(index) {}

		inline set<V*>& GetVertices() { return vertices; }
		inline set<T*>& GetTriangles() { return triangles; }

		bool operator==(const SpatialHashingCell& other) const
		{
			return get<0>(index) == get<0>(other.index) &&
				get<1>(index) == get<1>(other.index) &&
				get<2>(index) == get<2>(other.index);
		}

		bool operator!=(const SpatialHashingCell& other) const
		{
			return get<0>(index) != get<0>(other.index) ||
				get<1>(index) != get<1>(other.index) ||
				get<2>(index) != get<2>(other.index);
		}

	protected:
		tuple<int, int, int> index;

		set<V*> vertices;
		set<T*> triangles;
	};

	class SpatialHashing : public AABB, public ComponentBase {
	public:
		struct Vertex
		{
			size_t index = -1; // Vertex Buffer Index
		};

		struct Triangle
		{
			Vertex* v0 = nullptr;
			Vertex* v1 = nullptr;
			Vertex* v2 = nullptr;
		};

	public:
		SpatialHashing(const string& name, Mesh * mesh, float cellSize);
		~SpatialHashing();

		inline float GetCellSize() { return cellSize; }

		inline tuple<int, int, int> GetIndex(const glm::vec3& position)
		{
			auto x = (int)floorf((position.x + cellHalfSize) / cellSize);
			auto y = (int)floorf((position.y + cellHalfSize) / cellSize);
			auto z = (int)floorf((position.z + cellHalfSize) / cellSize);
			return make_tuple(x, y, z);
		}

		inline SpatialHashingCell<Vertex, Triangle>* GetCell(const tuple<int, int, int>& index)
		{
			if (cells.count(index) != 0) {
				return cells[index];
			}
			else {
				return nullptr;
			}
		}

		tuple<int, int, int> InsertPoint(const glm::vec3& position);

		tuple<int, int, int> InsertVertex(Vertex* vertex);
		void InsertTriangle(Triangle* t);

		void RemoveTriangle(Triangle* t);

		void Build();

		set<SpatialHashingCell<Vertex, Triangle>*> GetCellsWithRay(const Ray& ray);

		inline const map<tuple<int, int, int>, SpatialHashingCell<Vertex, Triangle>*>& GetCells() const { return cells; }

	private:
		Mesh* mesh = nullptr;
		float cellSize = 0.5;
		float cellHalfSize = 0.25;

		map<tuple<int, int, int>, SpatialHashingCell<Vertex, Triangle>*> cells;

		vector<Vertex> vertices;
		vector<Triangle> triangles;
	};

	typedef SpatialHashingCell<SpatialHashing::Vertex, SpatialHashing::Triangle> SHCell;
}
