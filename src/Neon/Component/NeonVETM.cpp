#include <Neon/Component/NeonVETM.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/SpatialPartitioning/NeonSpatialHashing.h>

namespace Neon
{
	//VETM::VETM(const string& name, Mesh* mesh)
	//	: ComponentBase(name), mesh(mesh)
	//{
	//}

	//VETM::~VETM()
	//{
	//}

	//void VETM::Clear()
	//{
	//	spatialHashing->Clear();

	//	//kdtree.Clear();

	//	//for (auto vertex : vertices)
	//	//{
	//	//	delete vertex;
	//	//}
	//	//vertices.clear();
	//	//vid = 0;

	//	//for (auto edge : edges)
	//	//{
	//	//	delete edge;
	//	//}
	//	//edges.clear();
	//	//eid = 0;
	//	//edgeMapping.clear();

	//	//for (auto triangle : triangles)
	//	//{
	//	//	delete triangle;
	//	//}
	//	//triangles.clear();
	//	//tid = 0;
	//	//triangleMapping.clear();

	//	//totalArea = 0.0;
	//}

	//void VETM::Clone(Mesh& clone)
	//{
	//	//clone.Clear();

	//	//map<Vertex*, Vertex*> vertexMapping;
	//	//for (auto& v : vertices)
	//	//{
	//	//	vertexMapping[v] = clone.AddVertex(v->p, { 0.0f, 0.0f, 0.0f, });
	//	//}

	//	//for (auto& t : triangles)
	//	//{
	//	//	clone.AddTriangle(vertexMapping[t->v0], vertexMapping[t->v1], vertexMapping[t->v2]);
	//	//}
	//}
}