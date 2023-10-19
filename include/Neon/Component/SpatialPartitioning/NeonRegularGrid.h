#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Mesh;

	template<typename V, typename T>
	class RegularGridCell : public AABB {
	public:
		RegularGridCell(const glm::vec3& minPoint = { FLT_MAX, FLT_MAX, FLT_MAX },
			const glm::vec3& maxPoint = { -FLT_MAX, -FLT_MAX, -FLT_MAX })
			: AABB(minPoint, maxPoint), index({ IntInfinity, IntInfinity, IntInfinity }) {}

		RegularGridCell(tuple<size_t, size_t, size_t> index,
			const glm::vec3& minPoint = { FLT_MAX, FLT_MAX, FLT_MAX },
			const glm::vec3& maxPoint = { -FLT_MAX, -FLT_MAX, -FLT_MAX })
			: AABB(minPoint, maxPoint), index(index) {}

		inline set<V*>& GetVertices() { return vertices; }
		inline set<T*>& GetTriangles() { return triangles; }

		bool operator==(const RegularGridCell& other) const
		{
			return get<0>(index) == get<0>(other.index) &&
				get<1>(index) == get<1>(other.index) &&
				get<2>(index) == get<2>(other.index);
		}

		bool operator!=(const RegularGridCell& other) const
		{
			return get<0>(index) != get<0>(other.index) ||
				get<1>(index) != get<1>(other.index) ||
				get<2>(index) != get<2>(other.index);
		}

	protected:
		tuple<size_t, size_t, size_t> index;

		set<V*> vertices;
		set<T*> triangles;
	};

	class RegularGrid : public AABB, public ComponentBase {
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
		RegularGrid(const string& name, Mesh* mesh, float cellSize);
		~RegularGrid();

		inline float GetCellSize() { return cellSize; }
		inline size_t GetCellCountX() const { return cellCountX; }
		inline size_t GetCellCountY() const { return cellCountY; }
		inline size_t GetCellCountZ() const { return cellCountZ; }

		inline tuple<size_t, size_t, size_t> GetIndex(const glm::vec3& position)
		{
			auto x = (size_t)floorf((position.x - this->GetMinPoint().x) / cellSize);
			auto y = (size_t)floorf((position.y - this->GetMinPoint().y) / cellSize);
			auto z = (size_t)floorf((position.z - this->GetMinPoint().z) / cellSize);
			return make_tuple(x, y, z);
		}

		inline RegularGridCell<Vertex, Triangle>* GetCell(const tuple<size_t, size_t, size_t>& index)
		{
			auto x = get<0>(index);
			auto y = get<1>(index);
			auto z = get<2>(index);

			if ((0 <= x && x < cellCountX) &&
				(0 <= y && y < cellCountY) &&
				(0 <= z && z < cellCountZ))
			{
				return cells[z][y][x];
			}
			else
			{
				return nullptr;
			}
		}

		void Build();

		tuple<size_t, size_t, size_t> InsertVertex(Vertex* vertex);
		void InsertTriangle(Triangle* t);

		inline const vector<Vertex*>& GetVertices() const { return vertices; }

	/*
		void RemoveTriangle(Triangle* t);

		set<RegularGridCell<Vertex, Triangle>*> GetCellsWithRay(const Ray& ray);
	*/

		inline const vector<vector<vector<RegularGridCell<Vertex, Triangle>*>>>& GetCells() const { return cells; }

	private:
		Mesh* mesh = nullptr;
		float cellSize = 0.5;
		float cellHalfSize = 0.25;
		size_t cellCountX = 0;
		size_t cellCountY = 0;
		size_t cellCountZ = 0;

		vector<vector<vector<RegularGridCell<Vertex, Triangle>*>>> cells;

		//map<tuple<int, int, int>, RegularGridCell<Vertex, Triangle>*> cells;

		vector<Vertex*> vertices;
		vector<Triangle*> triangles;
	};

	typedef RegularGridCell<RegularGrid::Vertex, RegularGrid::Triangle> RGCell;
}
