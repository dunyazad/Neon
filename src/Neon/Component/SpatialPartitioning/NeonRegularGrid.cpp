#include <Neon/Component/SpatialPartitioning/NeonRegularGrid.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/NeonVertexBufferObject.hpp>

namespace Neon
{
	RegularGrid::RegularGrid(const string& name, Mesh* mesh, float cellSize)
		: ComponentBase(name), mesh(mesh), cellSize(cellSize), cellHalfSize(cellSize * 0.5f)
	{
	}

	RegularGrid::~RegularGrid()
	{
		for (size_t z = 0; z < cellCountZ; z++)
		{
			for (size_t y = 0; y < cellCountY; y++)
			{
				for (size_t x = 0; x < cellCountX; x++)
				{
					SAFE_DELETE(cells[z][y][x]);
				}
			}
		}
		cells.clear();

		for (auto& v : vertices)
		{
			SAFE_DELETE(v);
		}
		vertices.clear();

		for (auto& t : triangles)
		{
			SAFE_DELETE(t);
		}
		triangles.clear();
	}

	void RegularGrid::Build()
	{
		auto meshAABB = mesh->GetAABB();
		cellCountX = (size_t)ceilf(meshAABB.GetXLength() / cellSize) + 1;
		cellCountY = (size_t)ceilf(meshAABB.GetYLength() / cellSize) + 1;
		cellCountZ = (size_t)ceilf(meshAABB.GetZLength() / cellSize) + 1;

		float nx = -((float)cellCountX * cellSize) * 0.5f;
		float px = ((float)cellCountX * cellSize) * 0.5f;
		float ny = -((float)cellCountY * cellSize) * 0.5f;
		float py = ((float)cellCountY * cellSize) * 0.5f;
		float nz = -((float)cellCountZ * cellSize) * 0.5f;
		float pz = ((float)cellCountZ * cellSize) * 0.5f;
		float cx = (px + nx) * 0.5f;
		float cy = (py + ny) * 0.5f;
		float cz = (pz + nz) * 0.5f;
		this->Expand(glm::vec3(nx, ny, nz) + meshAABB.GetCenter());
		this->Expand(glm::vec3(px, py, pz) + meshAABB.GetCenter());

		cells.resize(cellCountZ);
		for (size_t z = 0; z < cellCountZ; z++)
		{
			cells[z].resize(cellCountY);
			for (size_t y = 0; y < cellCountY; y++)
			{
				cells[z][y].resize(cellCountX);
				for (size_t x = 0; x < cellCountX; x++)
				{
					auto minPoint = glm::vec3(xyz.x + (float)x * cellSize, xyz.y + (float)y * cellSize, xyz.z + (float)z * cellSize);
					auto maxPoint = glm::vec3(xyz.x + (float)(x + 1) * cellSize, xyz.y + (float)(y + 1) * cellSize, xyz.z + (float)(z + 1) * cellSize);
					auto cell = new RegularGridCell<Vertex, Triangle>(minPoint, maxPoint);
					cells[z][y][x] = cell;
				}
			}
		}

		auto vb = mesh->GetVertexBuffer();
		auto nov = vb->Size();
		map<GLuint, Vertex*> indexVertexMapping;
		for (size_t i = 0; i < nov; i++)
		{
			auto v = new Vertex{ i };
			InsertVertex(v);
			indexVertexMapping[i] = v;
		}

		auto ib = mesh->GetIndexBuffer();
		auto noi = ib->Size();
		for (size_t i = 0; i < noi / 3; i++)
		{
			GLuint i0, i1, i2;
			mesh->GetTriangleVertexIndices(i, i0, i1, i2);
			auto t = new Triangle{ indexVertexMapping[i0], indexVertexMapping[i1], indexVertexMapping[i2] };
			InsertTriangle(t);
		}
	}

	tuple<size_t, size_t, size_t> RegularGrid::InsertVertex(Vertex* vertex)
	{
		auto vertexPosition = mesh->GetVertex(vertex->index);

		auto index = GetIndex(vertexPosition);
		auto cell = GetCell(index);
		if (nullptr != cell) {
			cell->GetVertices().insert(vertex);
		}
		return index;
	}

	void RegularGrid::InsertTriangle(Triangle* t)
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
		for (size_t z = get<2>(minIndex); z <= get<2>(maxIndex); z++) {
			for (size_t y = get<1>(minIndex); y <= get<1>(maxIndex); y++) {
				for (size_t x = get<0>(minIndex); x <= get<0>(maxIndex); x++) {
					auto cell = cells[(int)z][(int)y][(int)x];
					if (cell->IntersectsTriangle(p0, p1, p2))
					{
						cell->GetTriangles().insert(t);
					}
				}
			}
		}
	}

	//void RegularGrid::RemoveTriangle(Triangle* t)
	//{
	//	auto& p0 = mesh->GetVertex(t->v0->index);
	//	auto& p1 = mesh->GetVertex(t->v1->index);
	//	auto& p2 = mesh->GetVertex(t->v2->index);

	//	AABB taabb;
	//	taabb.Expand(p0);
	//	taabb.Expand(p1);
	//	taabb.Expand(p2);
	//	auto minIndex = GetIndex(taabb.GetMinPoint());
	//	auto maxIndex = GetIndex(taabb.GetMaxPoint());
	//	for (int z = get<2>(minIndex); z <= get<2>(maxIndex); z++) {
	//		for (int y = get<1>(minIndex); y <= get<1>(maxIndex); y++) {
	//			for (int x = get<0>(minIndex); x <= get<0>(maxIndex); x++) {
	//				auto cell = GetCell(make_tuple(x, y, z));
	//				if (nullptr != cell) {
	//					cell->GetTriangles().erase(t);
	//				}
	//			}
	//		}
	//	}
	//}

	//void RegularGrid::Build()
	//{
	//	auto nov = mesh->GetVertexBuffer()->Size();
	//	for (size_t i = 0; i < nov; i++)
	//	{
	//		vertices.push_back(Vertex{ i });

	//		InsertVertex(&(vertices[i]));
	//	}

	//	auto nof = mesh->GetIndexBuffer()->Size() / 3;
	//	for (size_t i = 0; i < nof; i++)
	//	{
	//		GLuint i0, i1, i2;
	//		mesh->GetTriangleVertexIndices(i, i0, i1, i2);

	//		triangles.push_back(Triangle{ &(vertices[i0]), &(vertices[i1]), &(vertices[i2]) });
	//		InsertTriangle(&(triangles[i]));
	//	}
	//}

	//set<RegularGridCell<RegularGrid::Vertex, RegularGrid::Triangle>*> RegularGrid::GetCellsWithRay(const Ray& ray)
	//{
	//	set<RegularGridCell<Vertex, Triangle>*> candidates;

	//	vector<glm::vec3> intersections;
	//	if (IntersectsRay(ray, intersections)) {
	//		auto origin = intersections[0];
	//		auto direction = glm::normalize(intersections[1] - intersections[0]);
	//		origin = origin - direction * cellSize * 2.0f;
	//		auto distance = glm::distance(intersections[0], intersections[1]) + cellSize * 2;
	//		auto accumulatedDistance = 0.0f;
	//		while (accumulatedDistance <= distance) {
	//			auto position = origin + direction * accumulatedDistance;
	//			auto index = GetIndex(position);
	//			for (int z = get<2>(index) - 1; z <= get<2>(index) + 1; ++z) {
	//				for (int y = get<1>(index) - 1; y <= get<1>(index) + 1; ++y) {
	//					for (int x = get<0>(index) - 1; x <= get<0>(index) + 1; ++x) {
	//						auto cell = GetCell(make_tuple(x, y, z));
	//						if (nullptr != cell) {
	//							candidates.insert(cell);
	//						}
	//					}
	//				}
	//			}
	//			accumulatedDistance += cellHalfSize;
	//		}
	//	}
	//	else {
	//		cout << "NOT intersects" << endl;
	//	}

	//	return candidates;
	//}
}
