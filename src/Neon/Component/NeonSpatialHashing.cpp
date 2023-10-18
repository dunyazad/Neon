#include <Neon/Component/NeonSpatialHashing.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/NeonVertexBufferObject.hpp>

namespace Neon
{
	SpatialHashing::SpatialHashing(const string& name, Mesh* mesh, float voxelSize)
		: ComponentBase(name), mesh(mesh), cellSize(voxelSize), cellHalfSize(voxelSize * 0.5f) {}

	SpatialHashing::~SpatialHashing()
	{
		for (auto& kvp : cells)
		{
			if (nullptr != kvp.second)
			{
				delete kvp.second;
				kvp.second = nullptr;
			}
		}
		cells.clear();
	}

	tuple<int, int, int> SpatialHashing::InsertPoint(const glm::vec3& position)
	{
		auto index = GetIndex(position);
		auto voxel = GetCell(index);
		if (nullptr == voxel) {
			auto minx = float(get<0>(index)) * cellSize - cellHalfSize;
			auto miny = float(get<1>(index)) * cellSize - cellHalfSize;
			auto minz = float(get<2>(index)) * cellSize - cellHalfSize;
			auto maxx = float(get<0>(index)) * cellSize + cellHalfSize;
			auto maxy = float(get<1>(index)) * cellSize + cellHalfSize;
			auto maxz = float(get<2>(index)) * cellSize + cellHalfSize;
			auto newVoxel = new SpatialHashingCell<Vertex, Triangle>(index, glm::vec3(minx, miny, minz), glm::vec3(maxx, maxy, maxz));
			cells[index] = newVoxel;
			Expand(position);
			return index;
		}
		else {
			return index;
		}
	}

	tuple<int, int, int> SpatialHashing::InsertVertex(Vertex* vertex)
	{
		auto vertexPosition = mesh->GetVertex(vertex->index);

		auto index = GetIndex(vertexPosition);
		auto voxel = GetCell(index);
		if (nullptr == voxel) {
			auto minx = float(get<0>(index)) * cellSize - cellHalfSize;
			auto miny = float(get<1>(index)) * cellSize - cellHalfSize;
			auto minz = float(get<2>(index)) * cellSize - cellHalfSize;
			auto maxx = float(get<0>(index)) * cellSize + cellHalfSize;
			auto maxy = float(get<1>(index)) * cellSize + cellHalfSize;
			auto maxz = float(get<2>(index)) * cellSize + cellHalfSize;
			auto newVoxel = new SpatialHashingCell<Vertex, Triangle>(index, glm::vec3(minx, miny, minz), glm::vec3(maxx, maxy, maxz));
			newVoxel->GetVertices().insert(vertex);
			cells[index] = newVoxel;
			Expand(vertexPosition);
			return index;
		}
		else {
			voxel->GetVertices().insert(vertex);
			return index;
		}
	}

	void SpatialHashing::InsertTriangle(Triangle* t)
	{
		auto& p0 = mesh->GetVertex(t->v0->index);
		auto& p1 = mesh->GetVertex(t->v1->index);
		auto& p2 = mesh->GetVertex(t->v2->index);

		auto normal = glm::normalize(glm::cross(glm::normalize(p1 - p0), glm::normalize(p2 - p0)));
		AABB taabb;
		taabb.Expand(p0);
		taabb.Expand(p1);
		taabb.Expand(p2);
		auto minIndex = GetIndex(taabb.GetMinPoint());
		auto maxIndex = GetIndex(taabb.GetMaxPoint());
		for (int z = get<2>(minIndex); z <= get<2>(maxIndex); z++) {
			for (int y = get<1>(minIndex); y <= get<1>(maxIndex); y++) {
				for (int x = get<0>(minIndex); x <= get<0>(maxIndex); x++) {
					float minx = float(x) * cellSize - cellHalfSize;
					float miny = float(y) * cellSize - cellHalfSize;
					float minz = float(z) * cellSize - cellHalfSize;
					float maxx = float(x) * cellSize + cellHalfSize;
					float maxy = float(y) * cellSize + cellHalfSize;
					float maxz = float(z) * cellSize + cellHalfSize;
					AABB aabb;
					aabb.Expand({ minx, miny, minz });
					aabb.Expand({ maxx, maxy, maxz });
					if (aabb.IntersectsTriangle(p0, p1, p2)) {
						auto index = make_tuple(x, y, z);
						if (cells.count(index) == 0) {
							auto newVoxel = new SpatialHashingCell<Vertex, Triangle>(index, aabb.GetMinPoint(), aabb.GetMaxPoint());
							newVoxel->GetTriangles().insert(t);
							cells[index] = newVoxel;
						}
						else {
							cells[index]->GetTriangles().insert(t);
						}
					}
				}
			}
		}
	}

	void SpatialHashing::RemoveTriangle(Triangle* t)
	{
		auto& p0 = mesh->GetVertex(t->v0->index);
		auto& p1 = mesh->GetVertex(t->v1->index);
		auto& p2 = mesh->GetVertex(t->v2->index);

		AABB taabb;
		taabb.Expand(p0);
		taabb.Expand(p1);
		taabb.Expand(p2);
		auto minIndex = GetIndex(taabb.GetMinPoint());
		auto maxIndex = GetIndex(taabb.GetMaxPoint());
		for (int z = get<2>(minIndex); z <= get<2>(maxIndex); z++) {
			for (int y = get<1>(minIndex); y <= get<1>(maxIndex); y++) {
				for (int x = get<0>(minIndex); x <= get<0>(maxIndex); x++) {
					auto voxel = GetCell(make_tuple(x, y, z));
					if (nullptr != voxel) {
						voxel->GetTriangles().erase(t);
					}
				}
			}
		}
	}

	void SpatialHashing::Build()
	{
		auto nov = mesh->GetVertexBuffer()->Size();
		for (size_t i = 0; i < nov; i++)
		{
			vertices.push_back(Vertex{ i });

			InsertVertex(&(vertices[i]));
		}

		auto nof = mesh->GetIndexBuffer()->Size() / 3;
		for (size_t i = 0; i < nof; i++)
		{
			GLuint i0, i1, i2;
			mesh->GetTriangleVertexIndices(i, i0, i1, i2);
			
			triangles.push_back(Triangle{ &(vertices[i0]), &(vertices[i1]), &(vertices[i2]) });
			InsertTriangle(&(triangles[i]));
		}
	}

	set<SpatialHashingCell<SpatialHashing::Vertex, SpatialHashing::Triangle>*> SpatialHashing::GetCellsWithRay(const Ray& ray)
	{
		set<SpatialHashingCell<Vertex, Triangle>*> candidates;

		vector<glm::vec3> intersections;
		if (IntersectsRay(ray, intersections)) {
			auto origin = intersections[0];
			auto direction = glm::normalize(intersections[1] - intersections[0]);
			origin = origin - direction * cellSize * 2.0f;
			auto distance = glm::distance(intersections[0], intersections[1]) + cellSize * 2;
			auto accumulatedDistance = 0.0f;
			while (accumulatedDistance <= distance) {
				auto position = origin + direction * accumulatedDistance;
				auto index = GetIndex(position);
				for (int z = get<2>(index) - 1; z <= get<2>(index) + 1; ++z) {
					for (int y = get<1>(index) - 1; y <= get<1>(index) + 1; ++y) {
						for (int x = get<0>(index) - 1; x <= get<0>(index) + 1; ++x) {
							auto voxel = GetCell(make_tuple(x, y, z));
							if (nullptr != voxel) {
								candidates.insert(voxel);
							}
						}
					}
				}
				accumulatedDistance += cellHalfSize;
			}
		}
		else {
			cout << "NOT intersects" << endl;
		}

		return candidates;
	}
}
